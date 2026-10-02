# Copyright 2025 Wisu Suntoyo — Apache-2.0
"""RLCD (Reinforcement Learning for Calibrated Decisions) training loop.

Placeholder for Phase 2+ implementation. The scaffold defines the module
boundary and CLI entry point; the RL loop will be implemented after the
supervised distillation baseline is validated.

Planned approach:
    - Reward signal: proper_reward() from Laya's common.py (log score + spherical + RPS)
    - TD(lambda) targets for multi-turn conversation trajectories
    - Policy gradient over the decision head; backbone frozen or lightly unfrozen
    - Calibration loss term to maintain ECE <= 0.065 during RL fine-tuning
"""

import argparse
import logging

_log = logging.getLogger("rawit.train_rlcd")


def main():
    parser = argparse.ArgumentParser(description="Rawit RLCD training (stub)")
    parser.add_argument("--checkpoint", required=True, help="Distilled checkpoint to start from.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    _log.info("RLCD training is not yet implemented. Checkpoint: %s", args.checkpoint)
    raise NotImplementedError(
        "RLCD loop is a Phase 2+ deliverable. "
        "Run train_distill.py first to produce a supervised baseline."
    )


if __name__ == "__main__":
    main()
