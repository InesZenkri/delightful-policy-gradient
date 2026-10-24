"""Small NumPy playground comparing REINFORCE and a delight-gated policy gradient on bandits."""
from .bandit import (
    softmax, advantage_ema_update, pg_direction, dg_gate, dg_direction,
    whiten_chi, BanditEnv, run_agent,
)

__all__ = ["softmax", "advantage_ema_update", "pg_direction", "dg_gate", "dg_direction", "whiten_chi",
           "BanditEnv", "run_agent"]
