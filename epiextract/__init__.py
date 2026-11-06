"""EpiExtract: find observational studies on an epidemiologic question and pull their
study design, Table 1 descriptives, Table 2 effect estimates, and adjustment sets
from open-access full text into one structured dataset.

Stages built so far (see docs/wire_diagram.png):
    1. Draft topic file      (Gemma)            topic.py
    2. Look up MeSH          (NCBI E-utilities) topic.py
    3. Review topic          (human checkpoint) topic.py
    4. Build query           (Python)           search.py
    5. Search PubMed         (NCBI E-utilities) search.py
    6. Rank by meaning       (Sentence-BERT)    rank.py
    7. Screen abstracts      (Gemma)            screen.py
    8. Get full text         (Europe PMC; Unpaywall)          fulltext.py

Run it with:  python -m epiextract "Is childhood trauma associated with psychosis?"
"""

__version__ = "0.1.0"
