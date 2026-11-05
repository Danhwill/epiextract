# EpiExtract

EpiExtract is a tool in development that searches PubMed for observational studies on an epidemiologic question, reads the open-access full text, and extracts each study's design, descriptive statistics (Table 1), effect estimates (OR, RR, or HR with 95% CIs, Table 2), and adjustment sets into one structured dataset. It also checks each paper for retractions. This repository currently holds the unit tests for one of its atomic protocols: **Rank by meaning**, which uses [Sentence-BERT](https://www.sbert.net) to rank PubMed abstracts by how closely they match the research question.

## Contract

| | Type | Description |
|---|---|---|
| **Input (encode)** | `str` or `list[str]` | English text: a research question, exposure/outcome terms, abstracts |
| **Output (encode)** | `numpy.ndarray` (float32) | Shape `(384,)` or `(n, 384)`; each vector has length 1 |
| **Input (compare)** | two embedding arrays | Queries and documents from encode |
| **Output (compare)** | `torch.Tensor` (float32) | Cosine similarity, shape `(n_queries, n_documents)`, values from −1 to 1 |

```python
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
scores = model.similarity(query_emb, doc_emb)
```

Citable source: Reimers N, Gurevych I. Sentence-BERT: sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP 2019*:3982-3992. https://arxiv.org/abs/1908.10084

## Tested versions

- sentence-transformers 6.2.0
- torch 2.14.1
- transformers 5.20.2
- Model `sentence-transformers/all-MiniLM-L6-v2`, pinned to the commit in `model_revision.txt`
- Python 3.10 or later

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The model (about 90 MB) downloads from Hugging Face on the first run.

## Running the tests

```bash
pytest -v
```

Expected result: **52 passed, 3 xfailed** (one FutureWarning about a renamed method is expected).

## What the tests cover

- **Typical inputs:** output shape, data type, and unit length; identical results on repeated runs; cosine similarity in the right range; documented example scores from the Sentence-BERT docs; public health synonyms ranked correctly (for example, "vaping" → "electronic vapor product use", "high blood pressure" → "hypertension", "SES", "PTSD"), including survey-question wording.
- **Edge cases:** empty strings and lists, whitespace, long inputs truncated at 256 word pieces, statistical notation and symbols (β, ≥, ±, 95% CI), non-English text, ties.
- **Invalid inputs:** `None`, integers, lists containing `None`, and mismatched embedding sizes raise errors instead of returning a result.

## Known limitations

These are marked `xfail(strict=True)`, so the suite will fail loudly if a future release changes the behavior:

- **Shared words beat meaning for short queries.** "Heart attack" ranks "Heart failure" and "Panic attack" above "Acute myocardial infarction" (tested in two forms). EpiExtract therefore also searches with generated synonym terms rather than relying on Sentence-BERT alone.
- **Abbreviations lose their case.** The model lowercases input, so "ENDS" (electronic nicotine delivery systems) is read as the word "ends".
