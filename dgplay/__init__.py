"""Small NumPy playground comparing REINFORCE and a delight-gated policy gradient on bandits."""
from .bandit import (
    softmax, advantage_ema_update, pg_direction, dg_gate, dg_direction,
    whiten_chi, BanditEnv, run_agent,
)
from .analysis import (
    visit_coefficient, direction_bias, expected_direction, simulate_expected,
    stall_equilibrium,
)

__all__ = ["softmax", "advantage_ema_update", "pg_direction", "dg_gate", "dg_direction", "whiten_chi",
           "BanditEnv", "run_agent", "visit_coefficient", "direction_bias", "expected_direction",
           "simulate_expected", "stall_equilibrium"]
