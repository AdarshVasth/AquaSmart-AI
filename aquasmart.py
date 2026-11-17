"""Synthetic industrial softener simulation and breakthrough forecasting."""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean

LIMIT = 25.0
FEATURES = ("resin_remaining", "inlet_mg_l", "flow_m3_h", "outlet_mg_l")

@dataclass
class Plant:
    capacity: float = 22.0
    remaining: float = 22.0
    regeneration_left: int = 0
    regenerations: int = 0

    def step(self, inlet, flow):
        if self.regeneration_left:
            self.regeneration_left -= 1
            if self.regeneration_left == 0:
                self.remaining = self.capacity
            return {"mode": "regeneration", "resin_remaining": self.remaining,
                    "inlet_mg_l": inlet, "flow_m3_h": 0, "outlet_mg_l": None}

        remaining_fraction = self.remaining / self.capacity
        leakage = 0.012 + 0.45 * max(0, 1 - remaining_fraction / 0.36) ** 2
        outlet = round(inlet * leakage, 2)
        # mg/L × m³/h divided by 1,000 gives kg/h.
        self.remaining = max(0.0, self.remaining - (inlet - outlet) * flow / 1000)
        if outlet >= LIMIT:
            self.regenerations += 1
            self.regeneration_left = 3
        return {"mode": "service", "resin_remaining": self.remaining,
                "inlet_mg_l": inlet, "flow_m3_h": flow, "outlet_mg_l": outlet}


def simulate(seed, hours):
    rng = Random(seed)
    plant = Plant()
    for hour in range(hours):
        inlet = rng.randint(200, 360)
        flow = rng.randint(3, 7)
        yield {"hour": hour, **plant.step(inlet, flow)}


def examples(rows, horizon=4):
    for i, row in enumerate(rows[:-horizon]):
        if row["mode"] != "service" or row["outlet_mg_l"] >= LIMIT:
            continue
        upcoming = rows[i + 1:i + 1 + horizon]
        target = any(r["outlet_mg_l"] is not None and r["outlet_mg_l"] >= LIMIT
                     for r in upcoming)
        yield row, vector(row), target


def vector(row):
    scales = (22, 360, 7, LIMIT)
    return [row[name] / scale for name, scale in zip(FEATURES, scales)]


def fit(examples):
    # Small k-nearest-neighbors classifier using only the standard library.
    training = [(features, label) for _, features, label in examples]

    def predict(features):
        neighbors = sorted(training, key=lambda item: sum(
            (a - b) ** 2 for a, b in zip(item[0], features)))[:9]
        return mean(label for _, label in neighbors)
    return predict


def score(actual, predicted):
    tp = sum(a and p for a, p in zip(actual, predicted))
    fp = sum(not a and p for a, p in zip(actual, predicted))
    fn = sum(a and not p for a, p in zip(actual, predicted))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("telemetry.csv"))
    args = parser.parse_args()
    if args.hours < 30:
        parser.error("--hours must be at least 30")
    if args.seed in range(200, 206):
        parser.error("--seed must differ from the reserved training seeds 200–205")

    predict = fit(item for seed in range(200, 206)
                  for item in examples(list(simulate(seed, 180))))
    demo = list(simulate(args.seed, args.hours))
    actual, predicted, baseline = [], [], []
    for row, features, label in examples(demo):
        risk = predict(features)
        alert = risk >= 0.5
        row.update(breakthrough_risk=round(risk, 3), early_warning=int(alert))
        actual.append(label)
        predicted.append(alert)
        baseline.append(row["resin_remaining"] < 6)
    columns = ["hour", "mode", *FEATURES, "breakthrough_risk", "early_warning"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(demo)
    breakthroughs = sum(r["outlet_mg_l"] is not None and r["outlet_mg_l"] >= LIMIT
                        for r in demo)
    print(f"{len(demo)} hours | {breakthroughs} breakthroughs | {sum(predicted)} ML warnings")
    print(f"Held-out simulation F1: ML {score(actual, predicted):.3f} | "
          f"resin threshold {score(actual, baseline):.3f} | {sum(actual)} positive windows")
    print(f"Telemetry: {args.output}")


if __name__ == "__main__":
    main()
