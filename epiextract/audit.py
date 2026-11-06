"""The audit trail: a record of every step, every model reply, and every flag.

Each stage calls audit.step(name, summary, details) when it finishes:
  - summary  one line shown in the Steps table, e.g. "568 matching papers (175 in PMC
             open access)" or "kept top 10 of 100 (similarity 0.72 to 0.63)"
  - details  the evidence behind that line, shown under "details" in audit.html:
             the reviewed topic file, the full PubMed query and any filters dropped
             to get results, the PubMed IDs fetched, the text Sentence-BERT ranked
             papers against, or the count for each full-text status
Problems that don't stop the run (a quote not found in the abstract, a failed
download, a missing MeSH heading) are recorded with audit.flag(). Every Gemma call
is logged automatically by llm.ask_json(): the prompt, the raw reply, how long it
took, and whether the code checks accepted it. At the end the run writes
audit.json (machine-readable) and audit.html (for a person to review).
"""

import html
import json
import time
from datetime import datetime


class Audit:
    def __init__(self, question):
        self.question = question
        self.started = datetime.now()
        self.steps = []        # {"name", "summary", "details", "seconds"}
        self.model_calls = []  # {"step", "attempt", "prompt", "reply", "problem", "seconds"}
        self.flags = []        # {"step", "item", "message"}
        self._t = time.time()

    def step(self, name, summary, details=None):
        now = time.time()
        self.steps.append({"name": name, "summary": summary, "details": details or {},
                           "seconds": round(now - self._t, 1)})
        self._t = now
        print(f"  ✓ {name}: {summary}")

    def model_call(self, step, attempt, prompt, reply, problem, seconds):
        self.model_calls.append({"step": step, "attempt": attempt, "prompt": prompt,
                                 "reply": reply, "problem": problem, "seconds": round(seconds, 1)})

    def flag(self, step, item, message):
        self.flags.append({"step": step, "item": item, "message": message})

    def to_dict(self):
        return {"question": self.question, "started": self.started.isoformat(timespec="seconds"),
                "steps": self.steps, "flags": self.flags, "model_calls": self.model_calls}

    def save(self, folder, coverage=None, screened=None):
        (folder / "audit.json").write_text(json.dumps(self.to_dict(), indent=2, default=str),
                                           encoding="utf-8")
        (folder / "audit.html").write_text(self.render_html(coverage or [], screened or []),
                                           encoding="utf-8")

    # ------------------------------------------------------------------ HTML
    def render_html(self, coverage, screened):
        e = lambda s: html.escape(str(s))
        out = [f"<!doctype html><html><head><meta charset='utf-8'><title>EpiExtract audit</title>",
               "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1d2733}"
               "h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #ccd}"
               "table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd;padding:4px 6px;vertical-align:top;text-align:left}"
               "th{background:#eef3f9}.flag{color:#a33}.ok{color:#2a7a45}details{margin:4px 0}pre{white-space:pre-wrap;font-size:12px;background:#f6f7f9;padding:6px}"
               "</style></head><body>",
               f"<h1>EpiExtract audit</h1><p><b>Question:</b> {e(self.question)}<br>"
               f"<b>Run started:</b> {self.started:%Y-%m-%d %H:%M}</p>"]

        out.append("<h2>Steps</h2><table><tr><th>Step</th><th>Result</th><th>Seconds</th></tr>")
        for s in self.steps:
            det = ""
            if s["details"]:
                det = f"<details><summary>details</summary><pre>{e(json.dumps(s['details'], indent=2, default=str))}</pre></details>"
            out.append(f"<tr><td>{e(s['name'])}</td><td>{e(s['summary'])}{det}</td><td>{s['seconds']}</td></tr>")
        out.append("</table>")

        if coverage:
            counts = {}
            for c in coverage:
                counts[c["status"]] = counts.get(c["status"], 0) + 1
            out.append("<h2>Coverage report</h2><p>" + "; ".join(f"{e(k)}: {v}" for k, v in counts.items()) + "</p>")
            out.append("<table><tr><th>PMID</th><th>Title</th><th>Status</th><th>Where</th></tr>")
            for c in coverage:
                link = c.get("url") or ""
                where = f"<a href='{e(link)}'>{e(link)}</a>" if link else ""
                out.append(f"<tr><td><a href='https://pubmed.ncbi.nlm.nih.gov/{e(c['pmid'])}/'>{e(c['pmid'])}</a></td>"
                           f"<td>{e(c.get('title', ''))}</td><td>{e(c['status'])}</td><td>{where}</td></tr>")
            out.append("</table>")

        if screened:
            out.append("<h2>Abstract screening</h2><table><tr><th>Rank</th><th>PMID</th><th>Title</th>"
                       "<th>Included</th><th>Design</th><th>Reason / evidence</th></tr>")
            for r in screened:
                cls = "ok" if r["included"] else ""
                ver = "" if r.get("evidence_verified", True) else " <span class='flag'>(quote not found in abstract)</span>"
                out.append(f"<tr><td>{r['rank']}</td><td><a href='https://pubmed.ncbi.nlm.nih.gov/{e(r['pmid'])}/'>{e(r['pmid'])}</a></td>"
                           f"<td>{e(r['title'])}</td><td class='{cls}'>{'yes' if r['included'] else 'no'}</td>"
                           f"<td>{e(r['design'])}</td><td>{e(r['reason'])}<br><i>“{e(r['evidence'])}”</i>{ver}</td></tr>")
            out.append("</table>")

        out.append(f"<h2>Flags ({len(self.flags)})</h2>")
        if self.flags:
            out.append("<table><tr><th>Step</th><th>Item</th><th>Message</th></tr>")
            for f in self.flags:
                out.append(f"<tr><td>{e(f['step'])}</td><td>{e(f['item'])}</td><td class='flag'>{e(f['message'])}</td></tr>")
            out.append("</table>")
        else:
            out.append("<p>None.</p>")

        out.append(f"<h2>Model calls ({len(self.model_calls)})</h2>")
        for m in self.model_calls:
            status = f"<span class='flag'>problem: {e(m['problem'])}</span>" if m["problem"] else "<span class='ok'>accepted</span>"
            out.append(f"<details><summary>{e(m['step'])} · attempt {m['attempt']} · {m['seconds']} s · {status}</summary>"
                       f"<b>Prompt</b><pre>{e(m['prompt'])}</pre><b>Raw reply</b><pre>{e(m['reply'])}</pre></details>")
        out.append("</body></html>")
        return "\n".join(out)
