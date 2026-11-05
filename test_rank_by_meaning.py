"""Unit tests for the atomic protocol "Rank by meaning".

Protocol
    Rank by meaning: sentence embeddings and cosine similarity with
    Sentence-BERT (Python package sentence-transformers; model
    sentence-transformers/all-MiniLM-L6-v2), run on CPU.

Citable source
    Reimers N, Gurevych I. Sentence-BERT: sentence embeddings using Siamese
    BERT-networks. In: Proceedings of EMNLP-IJCNLP 2019. Association for
    Computational Linguistics; 2019:3982-3992. https://arxiv.org/abs/1908.10084
    Library: https://www.sbert.net, https://pypi.org/project/sentence-transformers/
    Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    (Apache-2.0), revision pinned to a Hugging Face commit hash on first run.

Input -> output contract
    Encode:  str or list[str] of English text
             -> numpy.ndarray of float32, shape (384,) for one str or (n, 384)
                for n strings; each row has L2 norm 1 (normalize_embeddings=True).
    Compare: two embedding arrays from Encode (queries, documents)
             -> torch.Tensor of float32, shape (n_queries, n_documents),
                cosine similarity in [-1, 1]; higher means more similar.

Configuration under test (the workflow's calls, nothing else)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                device="cpu", revision=MODEL_REVISION)
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    scores = model.similarity(query_emb, doc_emb)   # default cosine

Written for
    sentence-transformers==6.2.0, torch==2.14.1, transformers==5.20.2,
    huggingface-hub==2.3.0, numpy==2.5.4, pytest==9.1.1 (Python >= 3.10).

Model revision
    The commit hash is read from the MODEL_REVISION environment variable or
    from model_revision.txt next to this file. If neither exists, the first
    run looks up the Hub's current commit for the model and writes it to
    model_revision.txt; commit that file so later runs use the same weights.

The tests treat the protocol as a black box: they call only the documented
SentenceTransformer constructor, encode() and similarity().
"""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_REVISION_FILE = Path(__file__).with_name("model_revision.txt")


# --------------------------------------------------------------------------- #
# Shared fixtures and helpers (the model is loaded once per test session)
# --------------------------------------------------------------------------- #
def _model_revision() -> str:
    env = os.environ.get("MODEL_REVISION", "").strip()
    if env:
        return env
    if _REVISION_FILE.exists():
        return _REVISION_FILE.read_text(encoding="utf-8").strip()
    # First run only: record the Hub's current commit so later runs are pinned.
    from huggingface_hub import HfApi

    sha = HfApi().model_info(MODEL_ID).sha
    _REVISION_FILE.write_text(sha + "\n", encoding="utf-8")
    return sha


@pytest.fixture(scope="session")
def model_revision() -> str:
    return _model_revision()


@pytest.fixture(scope="session")
def model(model_revision: str) -> SentenceTransformer:
    return SentenceTransformer(MODEL_ID, device="cpu", revision=model_revision)


def enc(model: SentenceTransformer, texts):
    """The workflow's encode call, unchanged."""
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def scores(model: SentenceTransformer, queries, documents) -> torch.Tensor:
    """The workflow's compare call: similarity() with the default cosine."""
    return model.similarity(enc(model, queries), enc(model, documents))


def norms(e: np.ndarray) -> np.ndarray:
    return np.linalg.norm(e, axis=-1)


# --------------------------------------------------------------------------- #
# 1. Load, pin and offline reload
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "package, expected",
    list({"sentence-transformers": "6.2.0", "torch": "2.14.1", "transformers": "5.20.2"}.items()),
)
def test_installed_versions_match_pins(package, expected):
    # torch from the CPU index reports e.g. "2.14.1+cpu".
    assert version(package).split("+")[0] == expected


def test_model_loads_on_cpu_with_pinned_revision(model):
    assert model.device.type == "cpu"


def test_model_dimension_and_max_seq_length(model):
    assert model.get_sentence_embedding_dimension() == 384
    assert model.max_seq_length == 256


def test_offline_reload_matches_fixture(model, model_revision):
    text = "exposure to secondhand smoke"
    offline = SentenceTransformer(
        MODEL_ID, device="cpu", revision=model_revision, local_files_only=True
    )
    assert enc(offline, text) == pytest.approx(enc(model, text), abs=1e-6)


def test_unknown_model_id_raises():
    with pytest.raises((OSError, ValueError)):
        SentenceTransformer(
            "sentence-transformers/no-such-model-zz9-plural",
            device="cpu",
            local_files_only=True,
        )


# --------------------------------------------------------------------------- #
# 3. Encode (the workflow call)
# --------------------------------------------------------------------------- #
def test_encode_single_string_shape_dtype(model):
    e = enc(model, "body mass index")
    assert isinstance(e, np.ndarray)
    assert e.dtype == np.float32
    assert e.shape == (384,)


def test_encode_list_returns_float_array_n_by_384(model):
    e = enc(model, ["vaping", "asthma", "high school students"])
    assert isinstance(e, np.ndarray)
    assert e.dtype == np.float32
    assert e.shape == (3, 384)


def test_embeddings_have_unit_length(model):
    e = enc(model, ["Current electronic vapor product use", "asthma"])
    assert norms(e) == pytest.approx(1.0, abs=1e-5)


def test_same_input_gives_same_output(model):
    text = "asthma among high school students"
    assert enc(model, text) == pytest.approx(enc(model, text), abs=1e-6)


def test_encode_list_rows_match_single_calls(model):
    # 40 items crosses the default batch size of 32.
    texts = [f"synthetic survey topic {i}" for i in range(40)]
    batch = enc(model, texts)
    for i, text in enumerate(texts):
        assert batch[i] == pytest.approx(enc(model, text), abs=1e-5), texts[i]


def test_encode_duplicate_strings_identical_rows(model):
    e = enc(model, ["obesity", "obesity"])
    assert e[0] == pytest.approx(e[1], abs=1e-6)


def test_encode_empty_list(model):
    result = enc(model, [])
    assert np.asarray(result).shape[0] == 0


def test_empty_string_returns_valid_vector(model):
    e = enc(model, [""])
    assert e.shape == (1, 384)
    assert np.isfinite(e).all()
    assert norms(e) == pytest.approx(1.0, abs=1e-5)


def test_encode_whitespace_only(model):
    e = enc(model, ["   \t\n "])
    assert e.shape == (1, 384)
    assert np.isfinite(e).all()


def test_encode_abbreviation_case_folded(model):
    # The model is uncased, so the abbreviation "ENDS" is read as the word "ends".
    e = enc(model, ["ENDS", "ends"])
    assert e[0] == pytest.approx(e[1], abs=1e-6)


@pytest.mark.parametrize(
    "texts",
    [
        [
            "e-cigarette (ENDS) use ≥ 1 day/week",
            "β = 0.42 (95% CI 0.31–0.55), p < 0.001; age 15 ± 2 y",
        ]
    ],
)
def test_unusual_characters_are_handled(model, texts):
    e = enc(model, texts)
    assert e.shape == (2, 384)
    assert np.isfinite(e).all()
    assert norms(e) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    "texts",
    [["PM₂.₅ exposure (µg/m³)", "NO₂ and O₃ concentrations (ppb)", "HbA1c ≥ 6.5% (48 mmol/mol)"]],
)
def test_encode_unicode_subscripts_and_units(model, texts):
    e = enc(model, texts)
    assert e.shape == (3, 384)
    assert np.isfinite(e).all()
    assert norms(e) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    "texts",
    [["tabaquismo durante el embarazo", "吸烟与肺癌", "Rauchen in der Schwangerschaft"]],
)
def test_encode_non_english_text(model, texts):
    # Similarity is not asserted: the model is trained on English.
    e = enc(model, texts)
    assert e.shape == (3, 384)
    assert np.isfinite(e).all()
    assert norms(e) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# 4. Truncation at 256 word pieces
# --------------------------------------------------------------------------- #
def test_long_input_is_truncated_not_rejected(model):
    e = enc(model, "vaping " * 1000)
    assert e.shape == (384,)
    assert np.isfinite(e).all()
    assert float(np.linalg.norm(e)) == pytest.approx(1.0, abs=1e-5)


def test_encode_truncates_beyond_256_word_pieces(model):
    e = enc(
        model,
        ["the " * 300 + "smoking and lung cancer", "the " * 300 + "rainfall in coastal towns"],
    )
    # The first 256 word pieces are identical, so the embeddings match.
    assert float(e[0] @ e[1]) >= 0.9999


def test_encode_short_texts_not_truncated(model):
    e = enc(
        model,
        ["the " * 20 + "smoking and lung cancer", "the " * 20 + "rainfall in coastal towns"],
    )
    assert float(e[0] @ e[1]) < 0.99


# --------------------------------------------------------------------------- #
# 3. Encode: invalid inputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_input", [None])
def test_encode_none_raises(model, bad_input):
    with pytest.raises((TypeError, ValueError, AttributeError)):
        enc(model, bad_input)


@pytest.mark.parametrize("bad_input", [42])
def test_encode_integer_raises(model, bad_input):
    with pytest.raises((TypeError, ValueError)):
        enc(model, bad_input)


@pytest.mark.parametrize("bad_input", [["asthma", None]])
def test_encode_list_containing_none_raises(model, bad_input):
    with pytest.raises((TypeError, ValueError)):
        enc(model, bad_input)


# --------------------------------------------------------------------------- #
# 5. similarity() with the default cosine
# --------------------------------------------------------------------------- #
def test_similarity_matrix_shape_and_range(model):
    s = scores(model, ["vaping", "asthma"], ["e-cigarette use", "wheezing", "seat belt use"])
    assert isinstance(s, torch.Tensor)
    assert s.dtype == torch.float32
    assert tuple(s.shape) == (2, 3)
    assert float(s.min()) >= -1 - 1e-5
    assert float(s.max()) <= 1 + 1e-5


def test_similarity_default_fn_is_cosine(model):
    scores(model, ["vaping"], ["e-cigarette use"])
    assert model.similarity_fn_name == "cosine"


def test_similarity_matches_numpy_cosine(model):
    q = enc(model, ["vaping", "asthma"])
    d = enc(model, ["e-cigarette use", "wheezing", "seat belt use"])
    assert model.similarity(q, d).numpy() == pytest.approx(q @ d.T, abs=1e-5)


def test_similarity_identical_text_diagonal_and_symmetry(model):
    e = enc(model, ["vaping", "asthma", "seat belt use"])
    s = model.similarity(e, e).numpy()
    assert np.diag(s) == pytest.approx(1.0, abs=1e-4)
    assert s == pytest.approx(s.T, abs=1e-6)


def test_similarity_single_1d_query(model):
    q = enc(model, "smoking")
    assert q.shape == (384,)
    d = enc(model, ["tobacco use", "secondhand smoke exposure", "nicotine dependence"])
    assert tuple(model.similarity(q, d).shape) == (1, 3)


def test_similarity_empty_documents(model):
    q = enc(model, ["smoking", "obesity"])
    d = np.zeros((0, 384), dtype=np.float32)
    assert tuple(model.similarity(q, d).shape) == (2, 0)


def test_similarity_tie_identical_documents(model):
    s = scores(
        model,
        ["maternal smoking"],
        [
            "Cohort study of maternal smoking",
            "Cohort study of maternal smoking",
            "Cohort study of maternal alcohol use",
        ],
    )
    assert float(s[0, 0]) == pytest.approx(float(s[0, 1]), abs=1e-6)
    assert float(s[0, 0]) > float(s[0, 2])
    assert float(s[0, 1]) > float(s[0, 2])


# --------------------------------------------------------------------------- #
# 7. Published reference scores (sbert.net examples for this model)
# --------------------------------------------------------------------------- #
def test_similarity_documented_movie_example(model):
    s = scores(
        model,
        ["The new movie is awesome", "The cat sits outside", "A man is playing guitar"],
        ["The dog plays in the garden", "The new movie is so great", "A woman watches TV"],
    )
    assert float(s[0, 1]) == pytest.approx(0.8939, abs=0.01)
    assert float(s[0, 0]) == pytest.approx(0.0543, abs=0.01)
    assert int(s[0].argmax()) == 1


def test_similarity_documented_weather_example(model):
    texts = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]
    s = scores(model, texts, texts)
    assert float(s[0, 1]) == pytest.approx(0.6660, abs=0.01)
    assert float(s[0, 2]) == pytest.approx(0.1046, abs=0.01)
    assert float(s[1, 2]) == pytest.approx(0.1411, abs=0.01)


# --------------------------------------------------------------------------- #
# 5. similarity(): invalid inputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "a, b",
    [(np.zeros((2, 384), dtype=np.float32), np.zeros((3, 128), dtype=np.float32))],
)
def test_similarity_dimension_mismatch_raises(model, a, b):
    with pytest.raises(RuntimeError):
        model.similarity(a, b)


@pytest.mark.parametrize(
    "a, b",
    [(["not", "an", "embedding"], np.zeros((1, 384), dtype=np.float32))],
)
def test_similarity_non_array_input_raises(model, a, b):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        model.similarity(a, b)


# --------------------------------------------------------------------------- #
# 6. Semantic ranking
#    Every row below shares one check: the expected document scores highest.
#    Each row keeps its own test name from the plan; parametrize carries its data.
# --------------------------------------------------------------------------- #
def _assert_top1(model, query: str, documents: list[str], expected: int) -> None:
    s = scores(model, [query], documents)
    ranked = sorted(zip(s[0].tolist(), documents), reverse=True)
    assert int(s[0].argmax()) == expected, f"ranking for {query!r}: {ranked}"


def test_synonym_scores_higher_than_unrelated(model):
    s = scores(model, ["vaping"], ["Current electronic vapor product use", "Bicycle helmet use"])
    assert float(s[0, 0]) > float(s[0, 1])


@pytest.mark.parametrize(
    "query, documents, expected",
    [
        (
            "vaping",
            [
                "Current cigarette smoking",
                "Current electronic vapor product use",
                "Current smokeless tobacco use",
                "Secondhand smoke exposure at home",
            ],
            1,
        )
    ],
)
def test_rank_vaping_to_electronic_vapor_product(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


@pytest.mark.parametrize(
    "query, documents, expected",
    [("heart attack", ["Stroke", "Acute myocardial infarction", "Hypertension", "Type 2 diabetes"], 1)],
)
def test_rank_heart_attack_to_myocardial_infarction(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)

@pytest.mark.xfail(strict=True, reason="Known limitation: the model favors shared words ('heart', 'attack') over meaning, so 'Heart failure' outranks 'Acute myocardial infarction'")
@pytest.mark.parametrize(
    "query, documents, expected",
    [("heart attack", ["Heart failure", "Acute myocardial infarction", "Panic attack"], 1)],
)
def test_rank_synonym_beats_shared_word_distractors(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


@pytest.mark.parametrize(
    "query, documents, expected",
    [("high blood pressure", ["Hyperlipidemia", "Hypertension", "Hypotension", "Hyperglycemia"], 1)],
)
def test_rank_high_blood_pressure_to_hypertension(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


@pytest.mark.parametrize(
    "query, documents, expected",
    [
        (
            "flu shot",
            [
                "Pneumococcal vaccination ever",
                "Influenza vaccination in the past 12 months",
                "HPV vaccination series completed",
                "Tetanus booster in the past 10 years",
            ],
            1,
        )
    ],
)
def test_rank_flu_shot_to_influenza_vaccination(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


@pytest.mark.parametrize(
    "query, documents, expected",
    [
        (
            "binge drinking",
            [
                "Any alcohol use in the past 30 days",
                "Consumed 5 or more alcoholic beverages on one occasion",
                "Daily cigarette smoking",
                "Prescription opioid misuse",
            ],
            1,
        )
    ],
)
def test_rank_binge_drinking_to_definition(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known limitation: all-MiniLM-L6-v2 is uncased, so the abbreviation 'ENDS' "
        "is encoded as the common word 'ends' (see test_encode_abbreviation_case_folded). "
        "If this starts passing, remove the marker."
    ),
)
@pytest.mark.parametrize(
    "query, documents, expected",
    [
        (
            "ENDS",
            [
                "Combustible cigarette smoking",
                "Electronic nicotine delivery systems",
                "Nicotine replacement therapy",
                "Hookah smoking",
            ],
            1,
        )
    ],
)
def test_rank_abbreviation_ends(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


@pytest.mark.parametrize(
    "query, documents, expected",
    [
        (
            "SES",
            ["Educational attainment", "Socioeconomic status", "Health insurance coverage", "Food insecurity"],
            1,
        )
    ],
)
def test_rank_abbreviation_ses(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


@pytest.mark.parametrize(
    "query, documents, expected",
    [
        (
            "PTSD",
            [
                "Major depressive disorder",
                "Post-traumatic stress disorder",
                "Generalized anxiety disorder",
                "Traumatic brain injury",
            ],
            1,
        )
    ],
)
def test_rank_abbreviation_ptsd(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


@pytest.mark.parametrize(
    "query, documents, expected",
    [
        (
            "vaping",
            [
                "During the past 30 days, on how many days did you use an electronic vapor product?",
                "During the past 30 days, on how many days did you smoke cigarettes?",
                "During the past 30 days, on how many days did you have at least one drink of alcohol?",
                "During the past 30 days, on how many days did you use chewing tobacco, snuff, or dip?",
            ],
            0,
        )
    ],
)
def test_rank_survey_question_vaping(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


@pytest.mark.parametrize(
    "query, documents, expected",
    [
        (
            "depression",
            [
                "During the past 12 months, did you ever feel so sad or hopeless almost every day "
                "for two weeks or more in a row that you stopped doing some usual activities?",
                "During the past 12 months, did you ever seriously consider attempting suicide?",
                "On an average school night, how many hours of sleep do you get?",
                "During the past 12 months, how many times were you in a physical fight?",
            ],
            0,
        )
    ],
)
def test_rank_survey_question_depression(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


@pytest.mark.parametrize(
    "query, documents, expected",
    [
        (
            "exercise",
            [
                "During the past 7 days, on how many days were you physically active for a total "
                "of at least 60 minutes per day?",
                "On an average school day, how many hours do you watch TV?",
                "During the past 7 days, how many times did you eat fruit?",
                "On an average school night, how many hours of sleep do you get?",
            ],
            0,
        )
    ],
)
def test_rank_survey_question_exercise(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


@pytest.mark.xfail(strict=True, reason="Known limitation: the model favors shared words over meaning, so a question naming 'coronary heart disease' outranks the heart attack question")
@pytest.mark.parametrize(
    "query, documents, expected",
    [
        (
            "heart attack",
            [
                "Has a doctor ever told you that you had a myocardial infarction?",
                "Has a doctor ever told you that you had a stroke?",
                "Has a doctor ever told you that you had angina or coronary heart disease?",
                "Has a doctor ever told you that you had high blood pressure?",
            ],
            0,
        )
    ],
)
def test_rank_survey_question_heart_attack(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


def test_rank_dataset_titles_top3(model):
    documents = [
        "Fine particulate matter exposure and pediatric asthma emergency visits, 2015-2020",
        "Traffic-related air pollution and wheeze in school-age children",
        "Ozone levels and asthma exacerbations among children in urban counties",
        "Smoking prevalence among adults by county",
        "Weekly influenza hospitalizations",
        "Blood lead levels in children under 6",
        "Heat-related emergency department visits",
        "Childhood obesity prevalence by state",
        "Radon testing in homes",
        "Nitrate levels in community drinking water",
        "COVID-19 vaccination coverage by age group",
        "Dental caries among children aged 6-11",
    ]
    s = scores(model, ["air pollution and childhood asthma"], documents)
    assert set(torch.topk(s[0], 3).indices.tolist()) == {0, 1, 2}


@pytest.mark.parametrize(
    "query, documents, expected",
    [
        (
            "obesity",
            [
                "Body mass index ≥ 30 kg/m²",
                "Systolic blood pressure ≥ 140 mmHg",
                "Fasting plasma glucose ≥ 126 mg/dL",
                "LDL cholesterol ≥ 160 mg/dL",
            ],
            0,
        )
    ],
)
def test_rank_obesity_to_bmi_threshold(model, query, documents, expected):
    _assert_top1(model, query, documents, expected)


def test_statistical_notation_does_not_drive_score(model):
    s = scores(
        model,
        ["smoking and lung cancer"],
        [
            "Smoking and lung cancer (OR = 2.31, 95% CI 1.84–2.90, p < 0.001)",
            "Alcohol and liver cirrhosis (OR = 2.31, 95% CI 1.84–2.90, p < 0.001)",
        ],
    )
    assert float(s[0, 0]) > float(s[0, 1])


def test_numeric_thresholds_not_distinguished(model):
    # Characterisation test: the model largely ignores numbers, so the glue code
    # should not rely on it to tell thresholds apart.
    e = enc(model, ["Body mass index ≥ 30 kg/m²", "Body mass index < 18.5 kg/m²"])
    assert float(e[0] @ e[1]) >= 0.8
