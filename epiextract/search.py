"""Stages 4-5: build the PubMed query from the topic file and run the search.

HOW THE QUERY IS BUILT
Each concept (exposure, outcome) becomes one OR-block of its MeSH headings and its
label + synonyms as title/abstract phrases. The two blocks are joined with AND,
then the filters from topic.json are added (dates, English, humans, an
observational-design block, and exclusion of reviews/letters/case reports).
If the full query finds nothing, broaden() drops filters one at a time.
"""

import re

from lxml import etree

from . import web

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

DESIGN_BLOCK = ('("cohort studies"[MeSH Terms] OR "case-control studies"[MeSH Terms] OR '
                '"cross-sectional studies"[MeSH Terms] OR "longitudinal studies"[MeSH Terms] OR '
                'cohort[tiab] OR "case-control"[tiab] OR "cross-sectional"[tiab] OR longitudinal[tiab] OR '
                'prospective[tiab] OR "odds ratio"[tiab] OR "hazard ratio"[tiab] OR "relative risk"[tiab] OR '
                '"risk ratio"[tiab])')
EXCLUDE_TYPES = ('(review[pt] OR "systematic review"[pt] OR "meta-analysis"[pt] OR editorial[pt] OR '
                 'letter[pt] OR comment[pt] OR "case reports"[pt])')
ANIMALS_ONLY = '("animals"[MeSH Terms] NOT "humans"[MeSH Terms])'
OA_FILTER = '"pubmed pmc open access"[filter]'


def concept_block(c):
    parts = [f'"{h}"[MeSH Terms]' for h in c.get("mesh", [])]
    parts += [f'"{t}"[tiab]' for t in [c["label"]] + c.get("synonyms", [])]
    return "(" + " OR ".join(parts) + ")"


def build_query(topic, drop=()):
    """Return the PubMed query string. `drop` lists filter names to leave out."""
    f = topic.get("filters", {})
    q = f"{concept_block(topic['exposure'])} AND {concept_block(topic['outcome'])}"
    if f.get("observational_only") and "design" not in drop:
        q += f" AND {DESIGN_BLOCK}"
    if f.get("exclude_reviews") and "types" not in drop:
        q += f" NOT {EXCLUDE_TYPES}"
    if f.get("humans_only") and "humans" not in drop:
        q += f" NOT {ANIMALS_ONLY}"
    if f.get("english_only") and "english" not in drop:
        q += " AND english[la]"
    years = f.get("years")
    if years and "years" not in drop:
        q += f' AND ("{years[0]}"[dp] : "{years[1]}"[dp])'
    return q


def esearch(params):
    """Call esearch patiently. PubMed sometimes answers "Search Backend failed ...
    temporarily unavailable" inside an otherwise normal reply; those are retried on
    the same schedule as server errors (about 5 minutes), then reported as an outage."""
    url = f"{EUTILS}/esearch.fcgi"
    waits = web.PATIENT_WAITS
    for attempt in range(1, len(waits) + 2):
        r = web.get(url, {"db": "pubmed", "retmode": "json", **params}, ncbi=True, patient=True)
        res = r.json().get("esearchresult", {})
        error = res.get("ERROR", "")
        if not error:
            return res
        if "temporarily unavailable" not in error.lower() and "backend failed" not in error.lower():
            raise web.WebError(f"PubMed rejected the query: {error}")
        if attempt == len(waits) + 1:
            raise web.ServiceDown(web.outage_message(url, "search temporarily unavailable"))
        web.wait_and_say(url, "search temporarily unavailable", attempt, waits)


def count(query):
    res = esearch({"term": query, "retmax": 0})
    return int(res.get("count", 0)), res.get("warninglist", {})


def broaden(topic, audit):
    """Find a query that returns results, dropping filters one at a time if needed.
    Returns (query, n_hits, dropped_filters)."""
    ladder = [(), ("design",), ("design", "english"), ("design", "english", "years")]
    for drop in ladder:
        q = build_query(topic, drop)
        n, warnings = count(q)
        not_found = warnings.get("phrasesnotfound", [])
        if not_found:
            audit.flag("Build query", "phrases", f"PubMed did not recognize: {', '.join(not_found)}")
        if n > 0:
            return q, n, list(drop)
        audit.flag("Build query", "0 hits", f"no results with filters minus {list(drop) or 'nothing'}; broadening")
    return q, 0, list(drop)


def search(query, max_records):
    """Run the search; return PMIDs in PubMed's relevance order."""
    return esearch({"term": query, "retmax": max_records, "sort": "relevance"}).get("idlist", [])


def _text(el):
    return re.sub(r"\s+", " ", el.xpath("string()")).strip() if el is not None else ""


def parse_pubmed_xml(xml_bytes):
    """Turn efetch XML into a list of record dicts."""
    root = etree.fromstring(xml_bytes)
    records = []
    for art in root.findall(".//PubmedArticle"):
        ids = {i.get("IdType"): (i.text or "").strip()
               for i in art.findall("./PubmedData/ArticleIdList/ArticleId")}
        abstract = []
        for part in art.findall(".//Abstract/AbstractText"):
            label, text = part.get("Label"), _text(part)
            abstract.append(f"{label}: {text}" if label else text)
        year = art.findtext(".//JournalIssue/PubDate/Year") or \
            (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]
        records.append({
            "pmid": art.findtext(".//MedlineCitation/PMID"),
            "title": _text(art.find(".//ArticleTitle")),
            "abstract": " ".join(abstract),
            "journal": art.findtext(".//Journal/ISOAbbreviation") or art.findtext(".//Journal/Title") or "",
            "year": year,
            "doi": ids.get("doi", ""),
            "pmcid": ids.get("pmc", ""),
            "pub_types": [p.text for p in art.findall(".//PublicationType")],
        })
    return records


def fetch(pmids, batch=100):
    records = []
    for i in range(0, len(pmids), batch):
        chunk = pmids[i:i + batch]
        r = web.get(f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"},
                    ncbi=True, patient=True)
        records += parse_pubmed_xml(r.content)
    order = {p: n for n, p in enumerate(pmids)}
    return sorted(records, key=lambda rec: order.get(rec["pmid"], 1e9))


def run(topic, audit, max_records=200, oa_only=False):
    query, n, dropped = broaden(topic, audit)
    n_oa, _ = count(f"({query}) AND {OA_FILTER}") if n else (0, {})
    if oa_only:
        query = f"({query}) AND {OA_FILTER}"
        n = n_oa
    audit.step("Build query", f"{n} matching papers ({n_oa} in PMC open access)"
               + (f"; filters dropped to get results: {', '.join(dropped)}" if dropped else ""),
               {"query": query, "hits": n, "hits_open_access": n_oa, "dropped_filters": dropped})
    pmids = search(query, max_records) if n else []
    records = fetch(pmids)
    no_abstract = sum(1 for r in records if not r["abstract"])
    audit.step("Search PubMed", f"fetched {len(records)} records (top {max_records} by PubMed relevance); "
               f"{no_abstract} have no abstract", {"pmids": pmids})
    return query, {"hits": n, "hits_open_access": n_oa}, records
