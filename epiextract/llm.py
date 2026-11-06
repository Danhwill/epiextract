"""Calls to the local language model (Gemma 3 4B through Ollama).

Every Gemma step asks for JSON that matches a schema. Ollama's "structured
outputs" feature constrains the reply to that schema, and ask_json() still
checks the result and retries once, because a small model can return valid JSON
with an empty or nonsensical field.

Requires Ollama running locally (https://ollama.com) with the model pulled:
    ollama pull gemma3:4b
"""

import json
import os
import time

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("EPIEXTRACT_MODEL", "gemma3:4b")
TIMEOUT = 600   # seconds; a 4B model on a laptop CPU can be slow on long prompts


class LLMError(RuntimeError):
    """Gemma could not be reached, or did not return usable JSON after a retry."""


def check_ollama():
    """Fail early, with a clear message, if Ollama isn't running or the model is missing."""
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    except requests.RequestException as exc:
        raise LLMError(f"Ollama is not running at {OLLAMA_URL}. Open the Ollama app "
                       f"(or run `ollama serve`) and try again.") from exc
    names = {m.get("name", "") for m in tags.get("models", [])}
    if MODEL not in names and f"{MODEL}:latest" not in names:
        raise LLMError(f"Model {MODEL} is not installed. Run: ollama pull {MODEL}")


def _chat(system, prompt, schema):
    body = {
        "model": MODEL,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise LLMError(f"Ollama call failed: {exc}") from exc
    return r.json()["message"]["content"]


def ask_json(system, prompt, schema, check=None, audit=None, step=""):
    """Ask Gemma for JSON matching `schema`; return the parsed dict.

    check: optional function(dict) -> error message or None, for rules the
           schema can't express (e.g., "label must not be empty").
    audit: optional Audit; every attempt and raw reply is logged there.
    """
    note = ""
    for attempt in (1, 2):
        started = time.time()
        raw = _chat(system, prompt + note, schema)
        try:
            data = json.loads(raw)
            problem = None if isinstance(data, dict) else "reply was not a JSON object"
        except json.JSONDecodeError as exc:
            data, problem = None, f"invalid JSON ({exc})"
        if problem is None and check:
            problem = check(data)
        if audit:
            audit.model_call(step, attempt, prompt, raw, problem, time.time() - started)
        if problem is None:
            return data
        note = (f"\n\nYour previous reply could not be used: {problem}. "
                f"Reply again with JSON that follows the schema exactly.")
    raise LLMError(f"{step}: Gemma did not return usable JSON after 2 tries ({problem})")
