import unittest
from pathlib import Path

from finmodel.sft import load_config


class D0ConfigurationTests(unittest.TestCase):
    def test_predictor_and_decision_makers_share_the_dual_architecture(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        predictor = load_config(root / "d0_predictor.json")
        grpo = load_config(root / "d0_decision_grpo.json")
        ppo = load_config(root / "d0_decision_ppo_group.json")

        self.assertEqual(predictor["training"]["objective"], "dual_rank_return")
        self.assertEqual(predictor["training"]["selection_metric"], "exact_rank_ic")
        self.assertEqual(predictor["training"].get("top10_excess_weight", 0), 0)
        self.assertEqual(predictor["model"]["head_mode"], "dual")
        self.assertEqual(predictor["model"], grpo["model"])
        self.assertEqual(grpo["model"], ppo["model"])
        self.assertEqual(grpo["policy"], ppo["policy"])
        self.assertEqual(grpo["policy"]["action_mode"], "hysteresis")
        self.assertEqual(grpo["policy"]["return_feature_mode"], "proxy")
        for config in (grpo, ppo):
            self.assertEqual(
                config["backbone_checkpoint"],
                "artifacts/d0/predictor/best/model.pt",
            )

    def test_both_decision_routes_start_from_untrained_policy(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        grpo = load_config(root / "d0_decision_grpo.json")
        ppo = load_config(root / "d0_decision_ppo_group.json")
        self.assertEqual(grpo["seed"], ppo["seed"])
        self.assertIsNone(ppo["initial_policy_checkpoint"])
        self.assertFalse(ppo["training"]["backbone_trainable"])
        self.assertEqual(ppo["training"]["algorithm"], "ppo_gae_group")
        self.assertEqual(
            grpo["training"]["expected_world_size"] *
            grpo["training"]["rollouts_per_rank"],
            ppo["training"]["expected_world_size"] *
            ppo["training"]["rollouts_per_rank"],
        )


if __name__ == "__main__":
    unittest.main()
