"""Command line entry point: runs stages 1-8 and writes the run folder.

Usage
    python -m epiextract "Is childhood trauma associated with psychosis in adulthood?"
    python -m epiextract --topic runs/<earlier run>/topic.json     # reuse a reviewed topic
Options
    --yes            accept the drafted topic without the review pause
    --max-records N  papers fetched from PubMed (default 200)
    --top-n N        papers kept by Rank by meaning for screening (default 40)
    --oa-only        search only papers in PubMed Central's open-access subset
    --out DIR        where run folders go (default: runs/)

Each run creates  runs/<date_time>_<short question>/  containing:
    topic.json      the reviewed search strategy (reuse it with --topic)
    records.csv     every paper fetched from PubMed, with its similarity score
    screening.csv   the ranked papers with Gemma's decision, design, reason, and evidence
    coverage.csv    full-text status of each included paper
    xml/            open-access JATS XML for included papers
    audit.html      the human-readable audit (open it in a browser)
    audit.json      the same audit, machine-readable
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

from . import fulltext, llm, rank, screen, search, topic, web
from .audit import Audit


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("; ".join(v) if isinstance(v, list) else v) for k, v in row.items()})


def run_folder(base, question):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", question).strip("_")[:40] or "run"
    folder = Path(base) / f"{datetime.now():%Y-%m-%d_%H%M%S}_{slug}"
    folder.mkdir(parents=True)
    return folder


def main(argv=None):
    p = argparse.ArgumentParser(prog="epiextract", description="EpiExtract, stages 1-8.")
    p.add_argument("question", nargs="?", help="research question (not needed with --topic)")
    p.add_argument("--topic", help="path to a topic.json from an earlier run")
    p.add_argument("--yes", action="store_true", help="skip the topic review pause")
    p.add_argument("--max-records", type=int, default=200)
    p.add_argument("--top-n", type=int, default=40)
    p.add_argument("--oa-only", action="store_true")
    p.add_argument("--out", default="runs")
    args = p.parse_args(argv)
    if not args.question and not args.topic:
        p.error("give a research question, or --topic with a topic.json file")

    try:
        llm.check_ollama()
        if args.topic:
            t = topic.load_topic(Path(args.topic))
            problems = topic.validate_topic(t)
            if problems:
                p.error(f"{args.topic}: " + "; ".join(problems))
            question = args.question or t.get("question", "")
            t["question"] = question
        else:
            question = args.question
        audit = Audit(question)
        folder = run_folder(args.out, question)
        print(f"\nEpiExtract · {question}\n  run folder: {folder}\n")

        if args.topic:
            audit.step("Draft topic file", f"skipped; loaded {args.topic}", {"topic": t})
        else:
            t = topic.draft_topic(question, audit)
            t = topic.add_mesh(t, audit)
        t = topic.review_topic(t, folder / "topic.json", audit, assume_yes=args.yes)
        topic.save_topic(t, folder / "topic.json")

        query, hits, records = search.run(t, audit, args.max_records, args.oa_only)
        (folder / "query.txt").write_text(query + "\n", encoding="utf-8")
        ranked = rank.rank(t, records, audit, args.top_n)
        write_csv(folder / "records.csv", records,
                  ["rank", "similarity", "pmid", "year", "journal", "title", "doi", "pmcid", "pub_types"])
        screened = screen.screen(ranked, t, audit)
        write_csv(folder / "screening.csv", screened,
                  ["rank", "similarity", "pmid", "year", "title", "included", "design", "reason",
                   "evidence", "evidence_verified"])
        included = [r for r in screened if r["included"]]
        coverage = fulltext.retrieve(included, folder, audit)
        write_csv(folder / "coverage.csv", coverage,
                  ["pmid", "title", "doi", "pmcid", "status", "xml_file", "url", "oa_status", "version", "license"])
    except (llm.LLMError, web.WebError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        if "audit" in locals():
            audit.flag("run", "stopped", str(exc))
            audit.save(folder)
            print(f"  Partial audit saved in {folder}", file=sys.stderr)
            if (folder / "topic.json").exists():
                print(f"  Your reviewed topic is saved. To pick up from the search later, run:\n"
                      f"    python -m epiextract --topic \"{folder / 'topic.json'}\" --yes", file=sys.stderr)
        return 1

    audit.save(folder, coverage, screened)
    n_xml = sum(1 for c in coverage if c["status"] == "full_text_xml")
    print(f"\nDone. {len(included)} papers included; {n_xml} with full-text XML ready for table extraction.")
    audit_file = (folder / "audit.html").resolve()
    print(f"Open the audit in your browser: {audit_file.as_uri()}")
    print(f"  or run:  open \"{folder / 'audit.html'}\"   (Mac)")
    return 0
