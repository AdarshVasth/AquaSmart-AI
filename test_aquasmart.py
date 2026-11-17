import unittest

from aquasmart import LIMIT, Plant, examples, fit, simulate


class SimulatorTests(unittest.TestCase):
    def test_regeneration_restores_resin_and_stops_flow(self):
        plant = Plant(remaining=0.5)
        self.assertGreaterEqual(plant.step(300, 5)["outlet_mg_l"], LIMIT)
        for _ in range(3):
            row = plant.step(300, 5)
            self.assertEqual(row["mode"], "regeneration")
            self.assertEqual(row["flow_m3_h"], 0)
            self.assertIsNone(row["outlet_mg_l"])
        self.assertEqual(plant.remaining, plant.capacity)
        self.assertLess(plant.step(300, 5)["outlet_mg_l"], LIMIT)

    def test_forecast_labels_use_future_service_data(self):
        rows = list(simulate(42, 120))
        samples = list(examples(rows))
        self.assertTrue(any(label for _, _, label in samples))
        for row, _, label in samples:
            i = row["hour"]
            expected = any((r["outlet_mg_l"] or 0) >= LIMIT for r in rows[i + 1:i + 5])
            self.assertEqual(label, expected)

    def test_model_scores_held_out_samples(self):
        predict = fit(examples(list(simulate(200, 120))))
        risks = [predict(features) for _, features, _ in examples(list(simulate(42, 120)))]
        self.assertTrue(all(0 <= risk <= 1 for risk in risks))
        self.assertGreater(max(risks), min(risks))


if __name__ == "__main__":
    unittest.main()
