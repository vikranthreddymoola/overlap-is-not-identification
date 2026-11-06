#!/usr/bin/env bash
# Reruns every study and rebuilds the paper's numbers, tables and figures.
# Takes about 4 hours on 13 cores. Set N_JOBS to change the number of workers.
#
#   bash scripts/reproduce.sh            # everything
#   bash scripts/reproduce.sh --quick    # skip the simulations; rebuild from results/
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" != "--quick" ]]; then
    python scripts/download_data.py
    python experiments/prepare_data.py
    python experiments/prepare_data.py --dcwp

    python experiments/run_sims.py
    python experiments/run_sims.py --only s4checks,s4tuned
    python experiments/application.py
    python experiments/application.py --tuning
    python experiments/application.py --subsample
    python experiments/delivery.py
    python experiments/diagnostic.py
    python experiments/benchmark.py
    if python -c "import doubleml" 2>/dev/null; then
        python experiments/doubleml_check.py
    else
        echo "doubleml not installed; keeping results/doubleml_check.csv (see README)"
    fi
fi

python experiments/analysis.py
python experiments/figures.py
echo "done: paper/numbers.tex, paper/tab_*.tex and paper/figures/ are up to date"
