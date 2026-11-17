# AquaSmart AI

Industrial water-softener simulation with machine-learning warnings for hardness breakthrough. The project uses synthetic data and models one softener vessel.

## Run locally

Requires Python 3.10+ and no external packages. From this directory:

```sh
python aquasmart.py
python -m unittest -v
```

On Windows, use `py` if `python` is unavailable. Open `telemetry.csv` in Excel and plot inlet and outlet hardness by hour. Apply a filter for `early_warning = 1` to see the model warnings.

## How it works

1. The simulator varies inlet hardness and flow, reduces remaining resin capacity as hardness is removed, and raises outlet hardness as the resin depletes.
2. The simulated controller starts three hours of offline regeneration after an outlet reading reaches the demo threshold of 25 mg/L. The outlet reading is blank during regeneration.
3. A k-nearest-neighbors classifier uses current flow, hardness and remaining capacity to estimate whether the outlet will reach the demo threshold within four hours. The risk is the fraction of neighboring training samples with a breakthrough label.
4. The CLI evaluates the model against a simple remaining-capacity rule using a simulation seed excluded from training. Forecasts are advisory and the simulation controller uses its own rule.

Values and thresholds are illustrative. Performance on synthetic data does not establish performance on a real plant, and remaining capacity is directly observable in this simulator.

## Next experiments

- Try a new seed with `python aquasmart.py --seed 123 --output trial.csv`. Compare F1 and inspect when the earliest warning appears within each cycle.
- Simulate flow sensor drift and add a separate anomaly detector. Measure both missed anomalies and false alerts.
- After obtaining real data, evaluate on complete held-out operating cycles, estimate resin capacity from measurable signals, and compare warning lead times and false alarms per cycle.
- Add a scenario for demand swings and compare simulated regeneration strategies by downtime, hardness violations, and estimated salt use.
