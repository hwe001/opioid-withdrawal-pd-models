Online Resource 2: neonatal withdrawal data and model fits

Source: Mizuno T, McPhail BT, Kamatkar S, et al. (2021) Physiologic indirect
response modeling to describe buprenorphine pharmacodynamics in newborns treated
for neonatal opioid withdrawal syndrome. Clin Pharmacokinet 60:249-259,
https://doi.org/10.1007/s40262-020-00939-2 (Fig. 5, infants A and B).

Digitization: page rendered at 600 dpi; axes calibrated from tick marks by linear
regression (residual < 0.003 day); score markers found as connected components of
the marker colour (fragments merged within 16 px); concentration curve = mean
vertical position of the curve colour in each pixel column. The concentration
curve is the individual model prediction of the original authors, not measured
data; the three measured concentrations per infant are given separately.

Reproduce (in this order)
  python digitize_mizuno.py <Mizuno PDF> 600 dig600   (and 450 525 675 750 900)
  python digitization_error.py      -> digitization_error.json
  python fit_mizuno.py              -> fit_results.json (~15 min)
  python fit_mizuno.py MBimax       (adds the partial-agonist sensitivity model)
  python analyse_fits.py            -> summary, counterfactuals, Fig. 7
  python bootstrap_fits.py 200      -> bootstrap_results.json (~1 h on 8 cores)

Files
  digitize_mizuno.py   automatic digitization of Fig. 5 at a given resolution
  dig<dpi>_*.json      digitizations at six resolutions
  digitization_error.py / .json   between-resolution differences (precision)
  bootstrap_fits.py / bootstrap_results.json / bootstrap_output.txt
  finnegan_patient{A,B}.csv              12-h mean Finnegan scores
  concentration_patient{A,B}.csv         predicted blood concentration (ng/mL)
  observed_concentration_patient{A,B}.csv measured dried-blood-spot concentrations
  fit_mizuno.py     model definitions and fitting (python fit_mizuno.py; ~15 min)
  analyse_fits.py   summary table, counterfactuals and Fig. 7
  fit_results.json  fitted parameters and criteria
  fit_log*.txt, analysis_output.txt   console output of the runs
