#!/usr/bin/env python3
# pull every item from the inventory API and write one csv
import csv
import os
import sys

import requests

BASE = "https://inventory.example.internal/api/v2/items"
PAGE_SIZE = 200
TIMEOUT = 15  # seconds, the internal box is slow on Mondays
TOKEN_PATH = os.path.expanduser("~/.config/invapi/token")
FIELDS = ["id", "sku", "name", "qty_on_hand", "warehouse", "updated_at"]


def load_token():
    with open(TOKEN_PATH) as fh:
        return fh.read().strip()


def fetch_page(session, cursor=None):
    params = {"limit": PAGE_SIZE}
    if cursor:
        params["cursor"] = cursor
    resp = session.get(BASE, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def iter_items(session):
    cursor = None
    page_no = 0
    while True:
        payload = fetch_page(session, cursor)
        page_no += 1
        items = payload.get("items", [])
        print(f"page {page_no}: {len(items)} items", file=sys.stderr)
        for item in items:
            yield item
        cursor = payload.get("next_cursor")
        if not cursor:
            break


def main(out_path):
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {load_token()}"
    count = 0
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in iter_items(session):
            writer.writerow(item)
            count += 1
    print(f"wrote {count} rows to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "items.csv")
