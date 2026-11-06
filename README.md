# Overlap Is Not Identification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24000911.svg)](https://doi.org/10.5281/zenodo.24000911)

Code, simulation output and paper source for

> *Overlap Is Not Identification: Reactive Pricing, Exogenous Variation, and
> Double Machine Learning* (2026).

Read the paper: [paper/main.pdf](paper/main.pdf).

The repository has two parts:

- **`overlap_diag`**, a small Python package with a cross-fitted partially
  linear DML estimator and the diagnostics proposed in the paper: the
  overlap-inflation check, the tipping point and its benchmarks.
- **`experiments/`**, the scripts that produce every number, table and figure
  in the paper, with their stored output in `results/`.

All simulations use synthetic data. The two semi-synthetic studies use public
New York City data (Taxi and Limousine Commission trip records and Department
of Consumer and Worker Protection delivery-app statistics) with a simulated
outcome, so the true effect is known.

## Contents

- [The problem in one example](#the-problem-in-one-example)
- [Installation](#installation)
- [Using the package](#using-the-package)
- [Reproducing the paper](#reproducing-the-paper)
- [Where each result comes from](#where-each-result-comes-from)
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [Data sources](#data-sources)
- [Citation](#citation)
- [License](#license)

## The problem in one example

Platforms that price with algorithms raise pay where a forecast predicts a
shortfall. If the forecast uses information the analyst does not log, price
and outcome move together for reasons other than the effect of price. DML
studies often argue that their estimate is credible because overlap, the
share of price variance the controls do not explain, is high. The paper shows
that overlap adds up two things: exogenous variation, which identifies the
effect, and variation from confounders the controls miss, which biases it.
High overlap can therefore come with a badly biased estimate.

`examples/quickstart.py` simulates this. The true effect of pay is 0.5.

```
$ python examples/quickstart.py
controls X               theta =  0.005  95% CI [-0.013,  0.024]  overlap = 0.62
controls X + forecast    theta =  0.492  95% CI [ 0.470,  0.515]  overlap = 0.31

absorbed = 49.7%, z = -32.5, flag = True
tipping point of the coarse fit: 0.008
```

Without the forecast, overlap is 0.62 and the estimate is close to zero.
Logging the forecast halves overlap and recovers the effect. The
overlap-inflation check sees this from the two fits alone: the finer
adjustment absorbs half of the residual price variance and moves the estimate
by 32 standard errors, so the coarse estimate is flagged.

## Installation

Python 3.9 or later.

```bash
git clone https://github.com/vikranthreddymoola/overlap-is-not-identification.git
cd overlap-is-not-identification
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # exact versions used for the paper
pip install -e .
```

`requirements.txt` pins the versions used for the paper (Python 3.9.6 on
macOS, Apple silicon). Results reproduce to the last digit with these
versions. Other versions of scikit-learn or numpy can change the last digits
of individual estimates but not the conclusions.

On macOS, numpy 2.0 with Apple's Accelerate BLAS can print spurious
`RuntimeWarning: ... encountered in matmul` messages. The estimates are
unaffected; the experiment scripts and tests silence them.

To use only the package, without the experiments:

```bash
pip install .
```

## Using the package

### Estimating an effect

```python
from overlap_diag import plr

r = plr(y, d, X, n_folds=5, n_rep=1, seed=0, learner="spline")
r.theta, r.se, r.ci, r.pval
r.overlap      # Var(D - m_hat(X)) / Var(D), out of fold
```

`y` is the outcome, `d` the price or other continuous treatment, `X` the
controls. The same learner fits E[Y | X] and E[D | X] with K-fold cross-fitting.
With `n_rep > 1` the folds are redrawn and the estimate is the median over
splits; the variance adds the spread across splits.

Available learners (`overlap_diag.LEARNERS`):

| Name | Learner |
|---|---|
| `spline` | cubic B-splines per column (12 knots) and ridge regression with the penalty chosen by cross-validation; `spline:8` sets 8 knots |
| `spline_interaction` | the spline learner plus an `X3 * X4` column (Study 4) |
| `boosting` | scikit-learn `HistGradientBoostingRegressor` with default settings |
| `boosting_tuned` | learning rate 0.05, up to 2,000 trees, 63 leaves, early stopping |
| `boosting_cat` | 300 iterations, first two columns treated as categorical |
| `boosting_cat_tuned` | tuned settings, first two columns categorical |
| `boosting_cat3` | 300 iterations, first three columns categorical |
| `random_forest` | 300 trees, minimum leaf size 5, half the columns per split |

### The overlap-inflation check

Fit a coarse and a finer adjustment on the same rows with the same seed, then
compare them:

```python
from overlap_diag import overlap_inflation, plr

coarse = plr(y, d, X_coarse, seed=0)
fine = plr(y, d, X_fine, seed=0)
check = overlap_inflation(coarse, fine)

check.absorbed     # share of the coarse residual price variance the refinement removes
check.z            # shift in units of sqrt(se_coarse^2 + se_fine^2)
check.flag         # pre-specified rule: absorbed > 5% and |z| > 2
check.z_paired     # shift in units of the SE of the difference
check.flag_paired  # |z_paired| > 2
```

A refinement can be a finer calendar, a more flexible learner, or an extra
logged input such as the pricing rule's forecast. Under a constant effect, a
flag means the coarse residual contained confounded variation (Proposition 2
in the paper), and the coarse estimate should not be reported. If overlap
falls and the estimate does not move, the refinement only removed exogenous
variation.

The pre-specified rule is the primary test. The paired test has more power,
but it also responds to small changes in the learner's fit and, under
heterogeneous effects, to reweighting, so a paired flag on its own calls for a
closer look rather than a verdict. Both fits must use one split
(`n_rep=1`, the default) and the same seed so that their folds coincide.

### The tipping point and its benchmarks

```python
from overlap_diag import benchmark, tipping_point

tau = tipping_point(r, d)   # |theta| * sd(D - m_hat(X))
```

`tau` is the smallest outcome-scale effect of an unadjusted component that
could explain the estimate away. To judge whether a component that large is
plausible, compare it with groups of observed controls:

```python
rows = benchmark(y, d, X, groups={"calendar": [0, 1, 2], "demand": [3]}, seed=0)
```

Each row gives, for one group G, its outcome-side strength `k` and price-side
strength `t` beyond the other controls, the one-sided ratio `tau / k`, the
two-sided multiple `c_star` (how many times as strong as G, on both sides, an
unadjusted component would have to be to explain the estimate away), and the
estimate with G dropped. In the paper, `c_star` is below one for every group
whose omission moves the estimate by more than its own size, and the one-sided
ratio is a poor guide. Pass
`categorical=[True, False, ...]` to use boosting with categorical columns
instead of the spline learner.

## Reproducing the paper

### From the stored results (minutes)

`results/` holds the output of every run. To rebuild the numbers quoted in the
text (`paper/numbers.tex`), the tables (`paper/tab_*.tex`) and the figures
(`paper/figures/`):

```bash
bash scripts/reproduce.sh --quick
```

Then compile `paper/main.tex` with any LaTeX distribution
(`cd paper && pdflatex main && pdflatex main`, or `tectonic main.tex`).

### From scratch (about 4 hours on 13 cores)

```bash
bash scripts/reproduce.sh
```

This downloads the public data, rebuilds the inputs, reruns every study and
then rebuilds the numbers, tables and figures. The scripts use all cores but
one; set `N_JOBS` to change that (`N_JOBS=4 bash scripts/reproduce.sh`).

The steps, which can also be run one at a time from the repository root:

```bash
python scripts/download_data.py              # NYC TLC trips (522 MB) and the DCWP utilization table
python experiments/prepare_data.py           # trips -> zone-hour cells
python experiments/prepare_data.py --dcwp    # DCWP PDF -> CSV (needs PyMuPDF, below)

python experiments/run_sims.py                         # Studies 1-4
python experiments/run_sims.py --only s4checks,s4tuned # Study 4 checks
python experiments/application.py                      # rideshare study
python experiments/application.py --tuning             # learner check
python experiments/application.py --subsample          # subsample check
python experiments/delivery.py                         # delivery-style study
python experiments/diagnostic.py                       # overlap-inflation diagnostic
python experiments/benchmark.py                        # tipping-point benchmarks
python experiments/doubleml_check.py                   # DoubleML comparison (below)

python experiments/analysis.py               # paper/numbers.tex and paper/tab_*.tex
python experiments/figures.py                # paper/figures/
```

`run_sims.py --only` takes any of `s1,s2,s3,s4,s4checks,s4tuned`.

The DCWP table is published as a PDF. Extracting it needs PyMuPDF:

```bash
pip install pymupdf==1.26.5
```

### DoubleML comparison

`experiments/doubleml_check.py` re-estimates the key designs with the DoubleML
package and checks that both implementations agree. Install DoubleML in a
separate environment so that it cannot change the pinned versions:

```bash
python -m venv .venv-dml
.venv-dml/bin/pip install -r requirements.txt doubleml==0.10.1 -c requirements.txt
.venv-dml/bin/pip install -e .
.venv-dml/bin/python experiments/doubleml_check.py
```

With the same folds, the estimates and standard errors agree to machine
precision in all 1,350 runs. `scripts/reproduce.sh` runs this step only if
DoubleML is installed and otherwise keeps the stored
`results/doubleml_check.csv`.

### Seeds

Every replication draws its data from a fixed seed, so all results reproduce
exactly.

| Study | Data seed |
|---|---|
| Study 1 | 10000 + replication |
| Study 2 | 20000 + replication |
| Study 3 | 30000 + replication |
| Study 4 | 40000 + replication |
| Overlap-inflation diagnostic | 60000 + replication |
| Rideshare study, main run | 70000 + draw |
| Rideshare study, learner check | 80000 + draw |
| Rideshare study, subsample check | 91000 + draw (subsample: 90000 + draw) |
| Delivery-style study | 95000 + draw |

Within a study, cells share seeds across designs, so designs are compared on
the same data. Fold splits use the replication number plus the repetition
index.

### Runtimes

On a 14-core Apple silicon laptop with 13 worker processes:

| Run | Command | Time |
|---|---|---|
| Study 1 | `run_sims.py --only s1` | 19 min |
| Study 2 | `run_sims.py --only s2` | 116 min |
| Study 3 | `run_sims.py --only s3` | 27 min |
| Study 4 | `run_sims.py --only s4` | 11 min |
| Study 4 checks | `run_sims.py --only s4checks` | 16 min |
| Study 4, tuned learner at larger n | `run_sims.py --only s4tuned` | 3 min |
| Rideshare study | `application.py` | 6 min |
| Rideshare learner check | `application.py --tuning` | 10 min |
| Rideshare subsample check | `application.py --subsample` | 10 min |
| Delivery-style study | `delivery.py` | 2 min |
| Overlap-inflation diagnostic | `diagnostic.py` | 5 min |
| Tipping-point benchmarks | `benchmark.py` | 8 min |
| DoubleML comparison | `doubleml_check.py` | 10 min |

## Where each result comes from

| Paper | Script | Output |
|---|---|---|
| Sec. 5, Study 1 (reactive pricing), Fig. `reactive` | `run_sims.py --only s1` | `results/s1.csv` |
| Sec. 5, closed-form bias map, Fig. `bias_map` | `figures.py` | computed from `dgp.py` |
| Sec. 5, Study 2 (coarse calendar), Figs. `coarse_calendar`, `coverage` | `run_sims.py --only s2` | `results/s2.csv` |
| Sec. 5, Study 3 (usual checks), Table `tab_diagnostics`, Fig. `overlap_decomposition` | `run_sims.py --only s3` | `results/s3.csv` |
| Sec. 5, Study 4 (harder nuisance), Table `tab_harder`, Fig. `nuisance_error` | `run_sims.py --only s4,s4checks,s4tuned` | `results/s4*.csv` |
| Sec. 6, rideshare study, Table `tab_application`, Figs. `application_data`, `application_estimates` | `application.py` (all three modes) | `results/application*.csv`; `application_data` reads `data/zone_hour_2023-03.parquet` |
| Sec. 7, delivery-style study, Table `tab_delivery`, Fig. `delivery` | `delivery.py` | `results/delivery.csv` |
| Sec. 8, overlap-inflation diagnostic, Table `tab_diagnostic`, Fig. `diagnostic` | `diagnostic.py` | `results/diagnostic.csv` |
| Sec. 9, tipping-point benchmarks, Table `tab_benchmark` | `benchmark.py` | `results/benchmark.csv` |
| Sec. 10, DoubleML comparison | `doubleml_check.py` | `results/doubleml_check.csv` |

Every number in the text is a macro in `paper/numbers.tex`, written by
`experiments/analysis.py`; none is typed by hand.

## Repository layout

```
src/overlap_diag/
    dml.py            cross-fitted partially linear estimator and learners
    diagnostics.py    overlap-inflation check, tipping point, benchmarks
experiments/
    dgp.py            reactive-pricing and seasonal designs, with closed-form bias and overlap
    run_sims.py       Studies 1-4
    prepare_data.py   NYC trips -> zone-hour cells; DCWP PDF -> CSV
    application.py    semi-synthetic rideshare study
    delivery.py       delivery-style study calibrated to DCWP data
    diagnostic.py     evaluation of the overlap-inflation check
    benchmark.py      tipping-point benchmarks
    doubleml_check.py comparison with DoubleML
    analysis.py       numbers and tables for the paper
    figures.py        figures for the paper
scripts/
    download_data.py  fetches the public data and checks it against the paper's copies
    reproduce.sh      runs everything in order
examples/
    quickstart.py     the example above
tests/                unit tests and exact re-runs of stored replications
results/              output of every run (CSV)
paper/                the paper (main.pdf), LaTeX source, generated numbers and tables, figures
data/                 downloaded and prepared inputs (not under version control)
```

## Tests

```bash
pip install pytest
python -m pytest
```

The tests check the estimator and the diagnostics on small simulated designs
and re-run single replications of Studies 1 and 4 and of the diagnostic,
comparing them with the stored results. They take about 10 seconds.

## Data sources

| Data | Publisher | Used for |
|---|---|---|
| [High Volume For-Hire Vehicle Trip Records, March 2023](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | NYC Taxi and Limousine Commission | pay, demand, zones and calendar in the rideshare study |
| [Weekly utilization, delivery apps](https://www.nyc.gov/site/dca/workers/Delivery-Worker-Public-Hearing-Minimum-Pay-Rate.page) | NYC Department of Consumer and Worker Protection | the weekly utilization series in the delivery-style study |
| [Delivery worker pay reports](https://www.nyc.gov/site/dca/workers/Delivery-Worker-Public-Hearing-Minimum-Pay-Rate.page) | NYC Department of Consumer and Worker Protection | the average pay level ($21.63 per hour, restaurant apps, Q2 2026) |

The data files are not redistributed here. `scripts/download_data.py` fetches
them from the publishers and compares their SHA-256 hashes with the copies used
for the paper. If a publisher revises a file, the script reports the
difference and keeps the new file; results may then differ slightly. Only the
outcome is simulated; no data from any company is used.

## Citation

If you use this code, please cite it. `CITATION.cff` has the details, and
GitHub's "Cite this repository" button formats them. Every release is archived
on Zenodo; the DOI [10.5281/zenodo.24000911](https://doi.org/10.5281/zenodo.24000911) always points to the latest version.

```bibtex
@software{moola2026overlap,
  title   = {overlap-is-not-identification: code for Overlap Is Not
             Identification},
  author  = {Moola, Vikranth Reddy},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.24000911},
  url     = {https://github.com/vikranthreddymoola/overlap-is-not-identification}
}
```

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
