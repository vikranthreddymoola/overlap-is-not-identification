# data/

Input files for the semi-synthetic studies. Nothing here is under version
control. To fill it:

    python scripts/download_data.py
    python experiments/prepare_data.py          # -> zone_hour_2023-03.parquet
    python experiments/prepare_data.py --dcwp   # -> dcwp/weekly_utilization.csv

| File | Source |
|---|---|
| `fhvhv_tripdata_2023-03.parquet` (522 MB) | NYC Taxi and Limousine Commission, [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), High Volume For-Hire Vehicle trips, March 2023 |
| `zone_hour_2023-03.parquet` | built from the file above by `experiments/prepare_data.py` |
| `dcwp/Weekly-Utilization-Data-Delivery-Apps.pdf` | NYC Department of Consumer and Worker Protection, [delivery worker minimum pay rate page](https://www.nyc.gov/site/dca/workers/Delivery-Worker-Public-Hearing-Minimum-Pay-Rate.page) |
| `dcwp/weekly_utilization.csv` | extracted from the PDF by `experiments/prepare_data.py --dcwp` |
