"""Stage 7: Screen abstracts (Gemma), with code checks on what Gemma says.

HOW SCREENING WORKS
For each ranked paper, Gemma reads the title and abstract and answers: is this an
original study in humans that estimates the association between the exposure and
the outcome? It also guesses the study design and copies one sentence from the
abstract as evidence. Code then checks the answer:
  - the evidence sentence must really appear in the abstract (otherwise it is flagged
    as unverified; the decision stands but a person should look);
  - only observational designs are kept (trials and reviews are excluded);
  - PubMed's own publication type overrides Gemma if it says review/meta-analysis.
"""

import re

from . import llm

OBSERVATIONAL = ["cohort", "case-control", "cross-sectional", "other observational"]
DESIGNS = OBSERVATIONAL + ["randomized trial", "systematic review or meta-analysis",
                           "qualitative or descriptive only", "other"]
SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "design": {"type": "string", "enum": DESIGNS},
        "reason": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["relevant", "design", "reason", "evidence"],
}
SYSTEM = ("You screen abstracts for an epidemiology literature review. You answer strictly "
          "from the abstract and copy evidence word for word.")
PROMPT = """Exposure: {exposure} (also called: {exp_syn})
Outcome: {outcome} (also called: {out_syn}){population}

Title: {title}
Abstract: {abstract}

Answer in JSON:
- relevant: true only if this is an original study in humans that measures the exposure
  and the outcome and estimates the association between them (for example an odds ratio,
  hazard ratio, relative risk, or regression coefficient). False for reviews, protocols,
  commentaries, studies of a different exposure or outcome, and studies of a treatment.
- design: the study design.
- reason: one short sentence explaining your answer.
- evidence: one sentence copied exactly from the abstract that supports your answer."""

REVIEW_TYPES = {"Review", "Systematic Review", "Meta-Analysis"}


def _norm(s):
    s = s.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[‐-―−]", "-", s)            # all dash characters -> hyphen
    s = re.sub(r"[^a-z0-9%.,;:()'\"<>=+-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def quote_in_text(quote, text):
    """True if the quote (allowing '...' gaps) appears in the text, ignoring case,
    spacing, and dash/quote styles."""
    q = _norm(quote).strip(" .\"'")
    if len(q) < 15:
        return False
    body = _norm(text)
    return all(piece.strip(" .") in body for piece in re.split(r"\.\.\.|…", q) if piece.strip(" ."))


def _check(d):
    if not d.get("evidence", "").strip():
        return "evidence is empty"
    return None


def screen_one(rec, topic, audit):
    if not rec["abstract"]:
        audit.flag("Screen abstracts", rec["pmid"], "no abstract; not screened")
        return {"included": False, "design": "unknown", "reason": "no abstract",
                "evidence": "", "evidence_verified": False}
    pop = f"\nPopulation of interest: {topic['population']}" if topic.get("population") else ""
    prompt = PROMPT.format(
        exposure=topic["exposure"]["label"], exp_syn=", ".join(topic["exposure"]["synonyms"]) or "-",
        outcome=topic["outcome"]["label"], out_syn=", ".join(topic["outcome"]["synonyms"]) or "-",
        population=pop, title=rec["title"], abstract=rec["abstract"])
    d = llm.ask_json(SYSTEM, prompt, SCHEMA, check=_check, audit=audit,
                     step=f"Screen abstracts · PMID {rec['pmid']}")
    verified = quote_in_text(d["evidence"], rec["title"] + " " + rec["abstract"])
    if not verified:
        audit.flag("Screen abstracts", rec["pmid"], "evidence quote not found in the abstract")
    included, reason = bool(d["relevant"]), d["reason"].strip()
    if included and d["design"] not in OBSERVATIONAL:
        included = False
        reason += f" [code check: design '{d['design']}' is not observational; excluded]"
    if included and REVIEW_TYPES & set(rec.get("pub_types", [])):
        included = False
        reason += " [code check: PubMed lists this as a review; excluded]"
    return {"included": included, "design": d["design"], "reason": reason,
            "evidence": d["evidence"].strip(), "evidence_verified": verified}


def screen(records, topic, audit):
    """Screen each record; return all records with screening fields added."""
    for i, rec in enumerate(records, 1):
        print(f"    screening {i}/{len(records)}: PMID {rec['pmid']}", end="\r", flush=True)
        rec.update(screen_one(rec, topic, audit))
    print(" " * 60, end="\r")
    kept = [r for r in records if r["included"]]
    unverified = sum(1 for r in records if r["abstract"] and not r["evidence_verified"])
    designs = {}
    for r in kept:
        designs[r["design"]] = designs.get(r["design"], 0) + 1
    audit.step("Screen abstracts", f"included {len(kept)} of {len(records)}; designs {designs or '-'}; "
               f"{unverified} evidence quotes not found in the abstract")
    return records
