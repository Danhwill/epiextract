"""Unit tests for EpiExtract's own (disposable) components.

These run offline: no PubMed, no Gemma, no model download. They check the code
rules that sit between the atomic protocols: building the query, cleaning search
terms, validating topic files, verifying Gemma's evidence quotes, and parsing
PubMed records.
Run with:  python -m pytest tests -v
"""

from epiextract import screen, search, topic

TOPIC = {
    "question": "Is childhood trauma associated with psychosis?",
    "exposure": {"label": "childhood trauma", "synonyms": ["childhood abuse"],
                 "mesh": ["adverse childhood experiences"]},
    "outcome": {"label": "psychosis", "synonyms": [], "mesh": ["psychotic disorders"]},
    "population": "",
    "filters": dict(topic.DEFAULT_FILTERS, years=[2000, 2026]),
}


# ---------- query builder ----------

def test_concept_block_joins_mesh_and_text_words_with_or():
    block = search.concept_block(TOPIC["exposure"])
    assert block == ('("adverse childhood experiences"[MeSH Terms] OR "childhood trauma"[tiab] '
                     'OR "childhood abuse"[tiab])')


def test_query_includes_both_concepts_and_all_filters():
    q = search.build_query(TOPIC)
    assert '"childhood trauma"[tiab]' in q and '"psychotic disorders"[MeSH Terms]' in q
    assert search.DESIGN_BLOCK in q and search.EXCLUDE_TYPES in q
    assert "english[la]" in q and '("2000"[dp] : "2026"[dp])' in q


def test_broadening_drops_only_the_named_filters():
    q = search.build_query(TOPIC, drop=("design", "english"))
    assert search.DESIGN_BLOCK not in q and "english[la]" not in q
    assert search.EXCLUDE_TYPES in q and '"2000"[dp]' in q


def test_filters_turned_off_in_topic_are_left_out():
    t = dict(TOPIC, filters=dict(TOPIC["filters"], observational_only=False, english_only=False))
    q = search.build_query(t)
    assert search.DESIGN_BLOCK not in q and "english[la]" not in q


# ---------- search terms and topic files ----------

def test_clean_terms_removes_duplicates_label_long_and_unsafe_terms():
    terms = ["Childhood Trauma", "childhood abuse", "Childhood abuse", 'early "trauma"',
             "a b c d e f g", "  maltreatment. "]
    assert topic.clean_terms("childhood trauma", terms) == ["childhood abuse", "maltreatment"]


def test_validate_topic_accepts_a_good_topic():
    assert topic.validate_topic(TOPIC) == []


def test_validate_topic_reports_missing_label_and_bad_years():
    bad = dict(TOPIC, outcome={"label": " "}, filters={"years": [2000]})
    problems = topic.validate_topic(bad)
    assert any("outcome.label" in p for p in problems)
    assert any("years" in p for p in problems)


# ---------- evidence-quote check ----------

ABSTRACT = ("Childhood trauma was associated with a higher risk of psychosis "
            "(OR = 2.87, 95% CI 2.07–3.98) after adjustment for age and sex.")


def test_exact_quote_is_verified():
    assert screen.quote_in_text("Childhood trauma was associated with a higher risk of psychosis", ABSTRACT)


def test_quote_with_different_dash_and_case_is_verified():
    assert screen.quote_in_text("(or = 2.87, 95% ci 2.07-3.98)", ABSTRACT)


def test_quote_with_ellipsis_gap_is_verified():
    assert screen.quote_in_text("Childhood trauma was associated ... after adjustment for age and sex", ABSTRACT)


def test_invented_quote_is_not_verified():
    assert not screen.quote_in_text("Childhood trauma was not associated with psychosis", ABSTRACT)


def test_too_short_quote_is_not_verified():
    assert not screen.quote_in_text("psychosis", ABSTRACT)


# ---------- PubMed record parsing ----------

PUBMED_XML = b"""<?xml version="1.0"?><PubmedArticleSet><PubmedArticle>
<MedlineCitation><PMID>123</PMID><Article><Journal><JournalIssue><PubDate><Year>2021</Year></PubDate>
</JournalIssue><ISOAbbreviation>Schizophr Res</ISOAbbreviation></Journal>
<ArticleTitle>Trauma and <i>psychosis</i></ArticleTitle>
<Abstract><AbstractText Label="METHODS">A cohort.</AbstractText><AbstractText Label="RESULTS">OR 2.1.</AbstractText></Abstract>
<PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList></Article></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="pubmed">123</ArticleId><ArticleId IdType="doi">10.1/x</ArticleId>
<ArticleId IdType="pmc">PMC9</ArticleId></ArticleIdList>
<ReferenceList><Reference><ArticleIdList><ArticleId IdType="doi">10.9/other</ArticleId></ArticleIdList></Reference></ReferenceList>
</PubmedData></PubmedArticle></PubmedArticleSet>"""


def test_parse_pubmed_xml_reads_fields_and_ignores_reference_ids():
    [rec] = search.parse_pubmed_xml(PUBMED_XML)
    assert rec["pmid"] == "123" and rec["year"] == "2021" and rec["journal"] == "Schizophr Res"
    assert rec["title"] == "Trauma and psychosis"
    assert rec["abstract"] == "METHODS: A cohort. RESULTS: OR 2.1."
    assert rec["doi"] == "10.1/x" and rec["pmcid"] == "PMC9"   # not the reference's DOI
