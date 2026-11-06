"""Stage 8: Get full text (open-access JATS XML) and record coverage.

HOW FULL-TEXT RETRIEVAL WORKS
1. Europe PMC looks up each PubMed ID and returns its PubMed Central ID (PMCID), if any,
   and whether the paper is in the open-access subset (licensed for text mining).
   PubMed's own record also lists the PMCID, which is used if Europe PMC can't be reached.
2. For open-access papers, Europe PMC's fullTextXML service returns the article as
   structured JATS XML, saved to the run's xml/ folder for the table parser.
   PMC papers outside the open-access subset can be read online but not mined.
3. Papers without XML are checked with Unpaywall (by DOI) to see whether a legal free
   copy exists elsewhere. That copy is recorded for the coverage report, not parsed.
   Unpaywall requires your email address: set it with  export EPIEXTRACT_EMAIL=you@...

Every included paper ends with one status:
  full_text_xml        XML saved; tables can be extracted
  in_pmc_not_oa        in PubMed Central, but not licensed for text mining (read it online)
  xml_download_failed  open access, but the XML download failed (rerun later)
  free_copy_elsewhere  no XML, but Unpaywall found a legal free copy (usually a PDF)
  abstract_only        no open full text found
  not_checked          no XML, and Unpaywall was skipped (no email set or no DOI)
"""

import os

from . import web

EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UNPAYWALL = "https://api.unpaywall.org/v2"


def europepmc_lookup(pmids):
    """Return {pmid: {"pmcid", "doi", "is_oa"}} from Europe PMC (50 PubMed IDs per call)."""
    out = {}
    for i in range(0, len(pmids), 50):
        chunk = pmids[i:i + 50]
        query = "(" + " OR ".join(f"EXT_ID:{p}" for p in chunk) + ") AND SRC:MED"
        r = web.get(f"{EUROPEPMC}/search", {"query": query, "format": "json",
                                           "resultType": "lite", "pageSize": 100}, patient=True)
        for rec in r.json().get("resultList", {}).get("result", []):
            if rec.get("pmid"):
                out[str(rec["pmid"])] = {"pmcid": rec.get("pmcid") or "", "doi": rec.get("doi") or "",
                                         "is_oa": rec.get("isOpenAccess") == "Y"}
    return out


def fetch_xml(pmcid):
    """Return the JATS XML bytes for an open-access article, or None."""
    r = web.get(f"{EUROPEPMC}/{pmcid}/fullTextXML", allow_404=True)
    if r.status_code == 200 and b"<article" in r.content[:2000]:
        return r.content
    return None


def unpaywall(doi, email):
    r = web.get(f"{UNPAYWALL}/{doi}", {"email": email}, allow_404=True)
    if r.status_code == 404:
        return None
    d = r.json()
    best = d.get("best_oa_location") or {}
    return {"is_oa": bool(d.get("is_oa")), "oa_status": d.get("oa_status", ""),
            "url": best.get("url_for_pdf") or best.get("url") or "",
            "version": best.get("version", ""), "host_type": best.get("host_type", ""),
            "license": best.get("license") or ""}


def retrieve(records, folder, audit, email=None):
    """Download XML for included records; return one coverage row per record."""
    email = email or os.environ.get("EPIEXTRACT_EMAIL", "")
    xml_dir = folder / "xml"
    xml_dir.mkdir(exist_ok=True)
    try:
        ids = europepmc_lookup([r["pmid"] for r in records]) if records else {}
    except web.WebError as exc:
        # Fall back to the PMCIDs in the PubMed records; open-access status is then unknown,
        # so the XML download is simply tried for every paper that has a PMCID.
        audit.flag("Get full text", "Europe PMC lookup", f"failed ({exc}); using PubMed's PMC IDs")
        ids = {r["pmid"]: {"pmcid": r.get("pmcid", ""), "doi": r.get("doi", ""), "is_oa": None}
               for r in records}
    coverage = []
    for i, rec in enumerate(records, 1):
        print(f"    full text {i}/{len(records)}: PMID {rec['pmid']}", end="\r", flush=True)
        info = ids.get(rec["pmid"], {})
        pmcid = info.get("pmcid") or rec.get("pmcid", "")
        doi = rec.get("doi") or info.get("doi", "")
        row = {"pmid": rec["pmid"], "title": rec["title"], "doi": doi, "pmcid": pmcid,
               "status": "", "xml_file": "", "url": "", "oa_status": "", "version": "", "license": ""}
        oa = info.get("is_oa")
        xml = None
        if pmcid and oa is not False:
            try:
                xml = fetch_xml(pmcid)
            except web.WebError as exc:
                audit.flag("Get full text", rec["pmid"], f"open-access XML download failed ({exc})")
            if xml is None and oa:
                audit.flag("Get full text", rec["pmid"], f"{pmcid} is open access but no XML was returned")
        if xml:
            path = xml_dir / f"{pmcid}.xml"
            path.write_bytes(xml)
            row.update(status="full_text_xml", xml_file=str(path.relative_to(folder)),
                       url=f"https://europepmc.org/article/PMC/{pmcid}")
        elif pmcid:
            row.update(status="xml_download_failed" if oa else "in_pmc_not_oa",
                       url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/")
        elif doi and email:
            try:
                u = unpaywall(doi, email)
            except web.WebError as exc:
                audit.flag("Get full text", rec["pmid"], f"Unpaywall lookup failed ({exc})")
                u = None
            if u and u["is_oa"] and u["url"]:
                row.update(status="free_copy_elsewhere", url=u["url"], oa_status=u["oa_status"],
                           version=u["version"], license=u["license"])
            else:
                row["status"] = "abstract_only"
        else:
            row["status"] = "not_checked"
        coverage.append(row)
    print(" " * 60, end="\r")
    counts = {}
    for c in coverage:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    note = "" if email else " (Unpaywall skipped: set EPIEXTRACT_EMAIL to check for free copies elsewhere)"
    audit.step("Get full text", ", ".join(f"{k}: {v}" for k, v in counts.items()) + note if counts else "no papers", counts)
    return coverage
