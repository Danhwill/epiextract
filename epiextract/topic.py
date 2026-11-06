"""Stages 1-3: draft the topic file, look up MeSH terms, and let the user review it.

The topic file (topic.json) is the search strategy: exposure and outcome terms,
their MeSH headings, and the filters. Saving it means a search can be rerun
exactly, and switching to another research domain needs only a new topic file.
"""

import json
import re
from datetime import date

from . import llm, web

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

DEFAULT_FILTERS = {
    "years": [2000, date.today().year],
    "english_only": True,
    "humans_only": True,
    "observational_only": True,   # adds a design block (cohort, case-control, ...)
    "exclude_reviews": True,      # drops reviews, meta-analyses, letters, case reports
}

# ---------------------------------------------------------------- 1. draft (Gemma)
CONCEPT = {
    "type": "object",
    "properties": {"label": {"type": "string"},
                   "synonyms": {"type": "array", "items": {"type": "string"}}},
    "required": ["label", "synonyms"],
}
DRAFT_SCHEMA = {
    "type": "object",
    "properties": {"exposure": CONCEPT, "outcome": CONCEPT, "population": {"type": "string"}},
    "required": ["exposure", "outcome", "population"],
}
DRAFT_SYSTEM = ("You help epidemiologists set up PubMed searches. You identify the exposure "
                "and the outcome in a research question and list the words researchers use "
                "for them in article titles and abstracts.")
DRAFT_PROMPT = """Research question: {question}

Return JSON with:
- exposure.label: the exposure (risk factor) in 1-4 words.
- exposure.synonyms: 3 to 6 other terms authors use for the same exposure (1-4 words each).
- outcome.label: the outcome (disease or condition) in 1-4 words.
- outcome.synonyms: 3 to 6 other terms authors use for the same outcome (1-4 words each).
- population: the population named in the question, or "" if none is named.

Synonyms must mean the same thing as the label or be a common close variant. Do not
list broader categories, related but different conditions, or study designs."""


def _check_draft(d):
    for part in ("exposure", "outcome"):
        if not str(d[part].get("label", "")).strip():
            return f"{part}.label is empty"
    return None


def clean_terms(label, terms, limit=8):
    """Tidy a list of search terms: trim, drop duplicates, the label itself, and
    anything too long or containing characters that would break a PubMed query."""
    seen, out = {label.strip().lower()}, []
    for t in terms:
        t = re.sub(r"\s+", " ", str(t)).strip(" .;,")
        key = t.lower()
        if not t or key in seen or len(t.split()) > 6 or re.search(r'["\[\]()]', t):
            continue
        seen.add(key)
        out.append(t)
    return out[:limit]


def draft_topic(question, audit):
    d = llm.ask_json(DRAFT_SYSTEM, DRAFT_PROMPT.format(question=question), DRAFT_SCHEMA,
                     check=_check_draft, audit=audit, step="Draft topic file")
    topic = {"question": question, "population": d.get("population", "").strip()}
    for part in ("exposure", "outcome"):
        label = d[part]["label"].strip()
        topic[part] = {"label": label, "synonyms": clean_terms(label, d[part].get("synonyms", [])),
                       "mesh": []}
    topic["filters"] = dict(DEFAULT_FILTERS)
    audit.step("Draft topic file",
               f"exposure = {topic['exposure']['label']} ({len(topic['exposure']['synonyms'])} synonyms); "
               f"outcome = {topic['outcome']['label']} ({len(topic['outcome']['synonyms'])} synonyms)",
               {"draft": d})
    return topic


# ---------------------------------------------------------------- 2. MeSH (NCBI)
# HOW THE MeSH LOOKUP WORKS
# PubMed's own "automatic term mapping" knows which MeSH heading each everyday
# phrase belongs to. Searching the MeSH database for a phrase returns that mapping
# (e.g., "childhood trauma" -> "adverse childhood experiences"[MeSH Terms]).
# We reuse it instead of asking Gemma, so every heading is a real MeSH term.
def mesh_for(term):
    r = web.get(f"{EUTILS}/esearch.fcgi", {"db": "mesh", "term": term, "retmode": "json"}, ncbi=True)
    result = r.json().get("esearchresult", {})
    headings = []
    for tr in result.get("translationset", []):
        # Keep a mapping only if it covers the WHOLE phrase. When a phrase has no
        # heading, PubMed maps its single words instead ("early life trauma" ->
        # "life"), and those word-level headings would flood the search.
        if tr.get("from", "").strip().lower() != term.strip().lower():
            continue
        headings += re.findall(r'"([^"]+)"\[MeSH Terms\]', tr.get("to", ""))
    return headings


def add_mesh(topic, audit):
    found = {}
    for part in ("exposure", "outcome"):
        concept = topic[part]
        headings = []
        for term in [concept["label"]] + concept["synonyms"]:
            try:
                mapped = mesh_for(term)
            except web.WebError as exc:
                audit.flag("Look up MeSH", term, f"lookup failed, skipped ({exc})")
                continue
            for h in mapped:
                if h not in headings:
                    headings.append(h)
                    found[h] = term
        concept["mesh"] = headings
        if not headings:
            audit.flag("Look up MeSH", part, "no MeSH heading found; the search will use text words only")
    audit.step("Look up MeSH",
               f"exposure: {', '.join(topic['exposure']['mesh']) or 'none'}; "
               f"outcome: {', '.join(topic['outcome']['mesh']) or 'none'}",
               {"heading <- term it came from": found})
    return topic


# ---------------------------------------------------------------- 3. review (human)
def validate_topic(t):
    """Return a list of problems with a topic dict (empty list = OK)."""
    problems = []
    for part in ("exposure", "outcome"):
        c = t.get(part)
        if not isinstance(c, dict) or not str(c.get("label", "")).strip():
            problems.append(f"{part}.label is missing")
            continue
        for key in ("synonyms", "mesh"):
            if not isinstance(c.get(key, []), list) or not all(isinstance(x, str) for x in c.get(key, [])):
                problems.append(f"{part}.{key} must be a list of text items")
    years = t.get("filters", {}).get("years", [])
    if not (isinstance(years, list) and len(years) == 2 and all(isinstance(y, int) for y in years)):
        problems.append("filters.years must be two whole numbers, like [2000, 2026]")
    return problems


def show_topic(t):
    f = t.get("filters", {})
    print(f"\n  Question:   {t.get('question', '')}")
    for part in ("exposure", "outcome"):
        c = t[part]
        print(f"  {part.capitalize():<11} {c['label']}")
        print(f"    MeSH:     {'; '.join(c.get('mesh', [])) or '(none)'}")
        print(f"    synonyms: {'; '.join(c.get('synonyms', [])) or '(none)'}")
    print(f"  Population: {t.get('population') or '(not specified; used for ranking and screening only)'}")
    print(f"  Filters:    years {f.get('years')}, English only = {f.get('english_only')}, "
          f"humans only = {f.get('humans_only')}, observational designs only = {f.get('observational_only')}, "
          f"exclude reviews = {f.get('exclude_reviews')}\n")


def save_topic(t, path):
    path.write_text(json.dumps(t, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_topic(path):
    t = json.loads(path.read_text(encoding="utf-8"))
    t.setdefault("population", "")
    t.setdefault("filters", {})
    for k, v in DEFAULT_FILTERS.items():
        t["filters"].setdefault(k, v)
    for part in ("exposure", "outcome"):
        if isinstance(t.get(part), dict):
            t[part].setdefault("synonyms", [])
            t[part].setdefault("mesh", [])
    return t


def review_topic(topic, path, audit, assume_yes=False):
    """Human checkpoint: show the topic, let the user edit topic.json, reload it."""
    save_topic(topic, path)
    show_topic(topic)
    if assume_yes:
        audit.step("Review topic", "skipped (--yes); topic accepted as drafted", {"topic": topic})
        return topic
    print(f"  The topic file is saved at:\n    {path}\n"
          "  To change it, open that file, edit the terms, and save it.")
    while True:
        answer = input("  Press Enter to continue with this topic, or type q to quit: ").strip().lower()
        if answer == "q":
            raise SystemExit("Stopped at the topic review. Your topic file is saved; rerun with --topic to reuse it.")
        try:
            edited = load_topic(path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  The file can't be read as JSON ({exc}). Fix it and press Enter again.")
            continue
        problems = validate_topic(edited)
        if problems:
            print("  Please fix: " + "; ".join(problems))
            continue
        changed = edited != topic
        if changed:
            show_topic(edited)
        audit.step("Review topic", "user confirmed" + (" after editing the file" if changed else " without changes"),
                   {"topic": edited})
        return edited
