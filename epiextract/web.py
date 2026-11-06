"""Shared HTTP helpers for the atomic protocols (NCBI, Europe PMC, Unpaywall, Crossref).

All web calls go through get() so that every request has a timeout, a polite
User-Agent, a pause between NCBI calls, and a retry when a server is busy.
"""

import os
import time
from urllib.parse import urlparse

import requests

USER_AGENT = "EpiExtract/0.1 (EPID 695/895 course project; https://github.com/Danhwill/epiextract)"
TIMEOUT = 30          # seconds to wait for any one response

# HOW RETRIES WORK
# Busy servers sometimes answer 429 (too many requests) or 5xx (server error).
# Single errors are common and clear within seconds, so every call retries quickly.
# Calls the run can't continue without (the PubMed search itself) are "patient":
# after the quick retries they try once a minute for about 5 minutes, and then stop
# with a plain message that the service is down, rather than failing silently.
QUICK_WAITS = [2, 5]                          # seconds between tries
PATIENT_WAITS = [2, 5, 60, 60, 60, 60, 60]    # about 5 minutes in total

# NCBI allows 3 requests per second without an API key and 10 with one.
# A free key is optional: set it with  export NCBI_API_KEY=...
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
NCBI_PAUSE = 0.12 if NCBI_API_KEY else 0.36   # seconds between NCBI calls (< 3 per second)

# NCBI asks every program to send a tool name and a contact email with each request,
# so they can reach the developer before blocking traffic. Unpaywall requires an email.
# The address is read from the environment, never written into the code:
#     export EPIEXTRACT_EMAIL=you@example.com
CONTACT_EMAIL = os.environ.get("EPIEXTRACT_EMAIL", "")

SERVICE_NAMES = {
    "eutils.ncbi.nlm.nih.gov": "NCBI's PubMed API (E-utilities)",
    "pmc.ncbi.nlm.nih.gov": "PubMed Central",
    "www.ebi.ac.uk": "Europe PMC",
    "api.unpaywall.org": "Unpaywall",
    "api.crossref.org": "Crossref",
}

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last_ncbi_call = 0.0


class WebError(RuntimeError):
    """A web service could not be reached or returned an error."""


class ServiceDown(WebError):
    """A service kept failing for several minutes: an outage on its side."""


def service_name(url):
    host = urlparse(url).netloc
    return SERVICE_NAMES.get(host, host)


def outage_message(url, detail):
    return (f"{service_name(url)} is not responding ({detail}). This is an outage on the "
            f"service's side, not a problem with EpiExtract or your computer. Outages like "
            f"this happen periodically and usually clear within a few hours; please try "
            f"again later.")


def wait_and_say(url, detail, attempt, waits):
    """Sleep before the next try; tell the user when the wait is long."""
    delay = waits[attempt - 1]
    if delay >= 30:
        print(f"\n    {service_name(url)} isn't responding ({detail}). "
              f"Trying again in {delay} s (try {attempt + 1} of {len(waits) + 1})...", flush=True)
    time.sleep(delay)


def get(url, params=None, ncbi=False, allow_404=False, patient=False):
    """GET a URL and return the requests.Response.

    ncbi=True     spaces calls out to respect NCBI's rate limit and adds the API key
                  and tool name.
    allow_404     returns a 404 response instead of raising, for services where
                  "not found" is an ordinary answer (e.g., no full text).
    patient=True  keeps retrying for about 5 minutes before giving up (ServiceDown).
    """
    global _last_ncbi_call
    params = dict(params or {})
    if ncbi:
        params.setdefault("tool", "epiextract")
        if CONTACT_EMAIL:
            params.setdefault("email", CONTACT_EMAIL)
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
    waits = PATIENT_WAITS if patient else QUICK_WAITS
    for attempt in range(1, len(waits) + 2):
        last = attempt == len(waits) + 1
        if ncbi:
            pause = NCBI_PAUSE - (time.time() - _last_ncbi_call)
            if pause > 0:
                time.sleep(pause)
            _last_ncbi_call = time.time()
        try:
            r = _session.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException as exc:
            detail = f"connection failed: {type(exc).__name__}"
            if last:
                raise (ServiceDown if patient else WebError)(outage_message(url, detail)) from exc
            wait_and_say(url, detail, attempt, waits)
            continue
        if r.status_code == 404 and allow_404:
            return r
        if r.status_code == 429 or r.status_code >= 500:
            detail = f"error {r.status_code}"
            if last:
                raise (ServiceDown if patient else WebError)(outage_message(url, detail))
            wait_and_say(url, detail, attempt, waits)
            continue
        if r.status_code >= 400:
            raise WebError(f"{service_name(url)} answered {r.status_code}: {r.text[:200]}")
        return r
