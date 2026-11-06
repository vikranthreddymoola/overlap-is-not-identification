"""Builds the input files for the semi-synthetic studies.

python experiments/prepare_data.py          # NYC FHV trips -> zone-hour cells
python experiments/prepare_data.py --dcwp   # DCWP utilization PDF -> CSV (needs PyMuPDF)
"""

from __future__ import annotations

import os

import pandas as pd
import pyarrow.parquet as pq

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
COLS = [
    "request_datetime",
    "PULocationID",
    "trip_miles",
    "trip_time",
    "driver_pay",
    "base_passenger_fare",
]


def weekly_utilization() -> None:
    """Pulls the weekly utilization table out of the DCWP PDF. Source:
    https://www.nyc.gov/site/dca/workers/Delivery-Worker-Public-Hearing-Minimum-Pay-Rate.page
    """
    import re

    import fitz

    src = os.path.join(DATA, "dcwp", "Weekly-Utilization-Data-Delivery-Apps.pdf")
    with fitz.open(src) as doc:
        tok = "".join(p.get_text() for p in doc).split()
    rows = [
        (tok[i], float(tok[i + 1][:-1]))
        for i in range(len(tok) - 1)
        if re.match(r"\d+/\d+/\d{4}$", tok[i]) and tok[i + 1].endswith("%")
    ]
    pd.DataFrame(rows, columns=["week_start", "utilization_pct"]).to_csv(
        os.path.join(DATA, "dcwp", "weekly_utilization.csv"), index=False
    )


def main() -> None:
    t = pq.read_table(
        os.path.join(DATA, "fhvhv_tripdata_2023-03.parquet"), columns=COLS
    ).to_pandas()
    t = t[(t.trip_miles > 0.2) & (t.driver_pay > 0) & (t.trip_time > 60)]
    t["hour"] = t.request_datetime.dt.floor("h")
    g = (
        t.groupby(["PULocationID", "hour"])
        .agg(
            trips=("driver_pay", "size"),
            pay=("driver_pay", "sum"),
            miles=("trip_miles", "sum"),
            secs=("trip_time", "sum"),
            fare=("base_passenger_fare", "sum"),
        )
        .reset_index()
    )
    g = g[(g.hour >= "2023-03-01") & (g.hour < "2023-04-01")]
    g["pay_per_mile"] = g.pay / g.miles
    coverage = g.groupby("PULocationID").size()
    keep = coverage[coverage >= 0.95 * 744].index  # zones observed in at least 95% of hours
    g[g.PULocationID.isin(keep)].to_parquet(
        os.path.join(DATA, "zone_hour_2023-03.parquet"), index=False
    )


if __name__ == "__main__":
    import sys

    weekly_utilization() if "--dcwp" in sys.argv else main()
