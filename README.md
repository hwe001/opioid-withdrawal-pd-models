# Opioid tolerance and withdrawal: PK/PD model comparison

Code and data for the manuscript

> Ho H. *Tolerance is not dependence: a mathematical analysis of pharmacodynamic models of opioid tolerance and withdrawal.* Submitted to the Journal of Pharmacokinetics and Pharmacodynamics.

The paper compares four adaptive structures driven by identical pharmacokinetics and receptor occupancy (indirect response without adaptation, a precursor pool, a feedback moderator, receptor down-regulation), derives their steady states and conservation properties, and applies the structures to two digitized neonatal withdrawal courses.

## Contents

| Path | What it does | Run time |
|---|---|---|
| `code/opioid_tolerance_sim.py` | All simulations: writes Figs 1-6 (`.pdf`, `.png`, `.tif`) and prints the numbers behind the tables | about 2 min |
| `neonatal/` | Digitized neonatal data, fitting code, bootstrap and the neonatal analysis (Fig. 7) | see below |

## Run

```bash
pip install -r requirements.txt
python code/opioid_tolerance_sim.py        # Figs 1-6 and table values
cd neonatal
python analyse_fits.py                      # summary, counterfactuals and Fig. 7 (about 30 s)
```

`analyse_fits.py` reads the shipped `fit_results.json` and reproduces `analysis_output.txt`.

Re-running the full neonatal pipeline (in this order):

```bash
python digitize_mizuno.py <Mizuno PDF> 600 dig600   # also 450 525 675 750 900
python digitization_error.py                        # between-resolution precision
python fit_mizuno.py                                # about 15 min
python fit_mizuno.py MBimax                         # partial-agonist sensitivity model
python analyse_fits.py
python bootstrap_fits.py 200                        # about 1 h on 8 cores
```

The digitizer needs the source article PDF (not redistributed here) and `pdftoppm` from poppler. The digitized results at six resolutions are included, so the fits and bootstraps do not need the PDF.

## Data provenance

The neonatal data were digitized from Fig. 5 (infants A and B) of Mizuno T, McPhail BT, Kamatkar S, et al. *Physiologic indirect response modeling to describe buprenorphine pharmacodynamics in newborns treated for neonatal opioid withdrawal syndrome.* Clin Pharmacokinet 2021;60:249-259, [doi:10.1007/s40262-020-00939-2](https://doi.org/10.1007/s40262-020-00939-2). The original individual data were not available. The concentration curves are the original authors' individual model predictions, not measurements; the three measured dried-blood-spot concentrations per infant are in `neonatal/observed_concentration_patient{A,B}.csv`. Details of the digitization and its precision are in `neonatal/README.txt`.

## Licence and citation

Code and digitized data: MIT licence (see `LICENSE`). Please cite the article above; see also `CITATION.cff`.
