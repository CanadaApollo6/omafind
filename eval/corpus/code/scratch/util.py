"""Small helpers I keep copying between projects.

retry_get() wraps requests.get with exponential backoff.
"""
import logging
import random
import time

import requests

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}


class GaveUp(Exception):
    """Raised after the last attempt fails."""

    def __init__(self, url, attempts, last_error):
        super().__init__(f"gave up on {url} after {attempts} tries: {last_error}")
        self.url = url
        self.attempts = attempts
        self.last_error = last_error


def backoff_delay(attempt, base=0.5, cap=30.0):
    # exponential: 0.5, 1, 2, 4, ... capped at 30 seconds
    delay = min(cap, base * (2 ** attempt))
    # jitter so a pile of scripts doesn't retry in lockstep
    return delay * random.uniform(0.8, 1.2)


def retry_get(url, tries=5, timeout=10, session=None, **kwargs):
    """GET url, waiting longer after each failure. Returns the Response."""
    http = session or requests
    last_error = None
    for attempt in range(tries):
        try:
            resp = http.get(url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            last_error = repr(exc)
        else:
            if resp.status_code not in RETRY_STATUSES:
                # 4xx other than 429 is our fault, don't retry it
                resp.raise_for_status()
                return resp
            last_error = f"status {resp.status_code}"

        if attempt == tries - 1:
            break
        wait = backoff_delay(attempt)
        log.warning("attempt %d for %s failed (%s), sleeping %.1fs",
                    attempt + 1, url, last_error, wait)
        time.sleep(wait)
    raise GaveUp(url, tries, last_error)


def chunked(items, size):
    # yield successive slices, the last one may be short
    for i in range(0, len(items), size):
        yield items[i:i + size]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # manual smoke test against a status endpoint, not run in CI
    r = retry_get("https://httpbin.org/status/503", tries=3)
    print(r.status_code)
