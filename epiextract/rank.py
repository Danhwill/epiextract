"""Stage 6: Rank by meaning (Sentence-BERT).

HOW THE RANKING WORKS
Sentence-BERT turns the research question and each paper's title + abstract into
384-number "meaning vectors". Cosine similarity between the question's vector and
a paper's vector says how close in meaning they are (1 = same meaning). We keep the
top N papers for Gemma to screen, which saves time on a laptop.

Same configuration as the unit tests in test_rank_by_meaning.py: CPU, normalized
embeddings, default cosine similarity, and the model revision in model_revision.txt.
Note: the model reads at most 256 word pieces (~190 words), so long abstracts are
judged on their title and opening sentences.
"""

from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION_FILE = Path(__file__).resolve().parent.parent / "model_revision.txt"

_model = None


def load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # slow import, so only when needed
        revision = REVISION_FILE.read_text().strip() if REVISION_FILE.exists() else None
        _model = SentenceTransformer(MODEL_ID, device="cpu", revision=revision)
    return _model


def query_text(topic):
    """The text the papers are compared against: the question plus the key terms."""
    parts = [topic.get("question", ""), topic["exposure"]["label"], topic["outcome"]["label"]]
    if topic.get("population"):
        parts.append(topic["population"])
    return ". ".join(p for p in parts if p)


def rank(topic, records, audit, top_n=40):
    """Add a 'similarity' score and 'rank' to each record; return the top_n records."""
    if not records:
        audit.step("Rank by meaning", "no records to rank")
        return []
    model = load_model()
    docs = [f"{r['title']}. {r['abstract']}" for r in records]
    q = model.encode([query_text(topic)], normalize_embeddings=True, convert_to_numpy=True)
    d = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
    scores = model.similarity(q, d)[0].tolist()
    for r, s in zip(records, scores):
        r["similarity"] = round(float(s), 4)
    ordered = sorted(records, key=lambda r: r["similarity"], reverse=True)
    for i, r in enumerate(ordered, 1):
        r["rank"] = i
    kept = ordered[:top_n]
    cutoff = kept[-1]["similarity"] if kept else None
    audit.step("Rank by meaning", f"kept top {len(kept)} of {len(records)} "
               f"(similarity {kept[0]['similarity']:.2f} to {cutoff:.2f})",
               {"compared against": query_text(topic)})
    return kept
