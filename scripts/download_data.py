"""Downloads the public input files into data/ and checks them against the
SHA-256 hashes of the copies used for the paper.

    python scripts/download_data.py

Publishers occasionally revise files. If a hash does not match, the script
says so and keeps the file; results may then differ slightly from the paper.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

FILES = [
    # (path under data/, url, sha256 of the copy used for the paper)
    (
        "fhvhv_tripdata_2023-03.parquet",
        "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-03.parquet",
        "57e607736d4460050e823acb1af82f9a23ed36382b131217fa56597f4f7c0f5a",
    ),
    (
        "dcwp/Weekly-Utilization-Data-Delivery-Apps.pdf",
        "https://www.nyc.gov/assets/dca/downloads/pdf/workers/"
        "Weekly-Utilization-Data-Delivery-Apps.pdf",
        "d25201b4fc2699a953de68db511e70051f8f59530b3ae080396f602e7f38cff5",
    ),
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, url: str, expected: str) -> None:
    path = os.path.join(DATA, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"downloading {url}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as f:
            while block := r.read(1 << 20):
                f.write(block)
        os.replace(path + ".part", path)
    got = sha256(path)
    status = "ok" if got == expected else f"differs from the paper's copy ({got})"
    print(f"{name}: {status}")


def main() -> None:
    for name, url, expected in FILES:
        fetch(name, url, expected)


if __name__ == "__main__":
    main()
