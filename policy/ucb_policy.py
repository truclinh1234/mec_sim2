# =============================================================================
# policy/ucb_policy.py — Contextual Bandit với Upper Confidence Bound (LinUCB)
# =============================================================================
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Dict, List
import config as cfg
from policy.base_policy import BasePolicy

if TYPE_CHECKING:
    from env.task import Task


class ArmState:
    """Trạng thái LinUCB của 1 arm (Ridge regression online)."""

    def __init__(self, d: int, lam: float = 1.0):
        self.A: np.ndarray = lam * np.eye(d)   # d×d covariance
        self.b: np.ndarray = np.zeros(d)        # d reward accumulator
        self._A_inv: np.ndarray | None = None   # cache

    def update(self, phi: np.ndarray, reward: float):
        self.A += np.outer(phi, phi)
        self.b += reward * phi
        self._A_inv = None                      # invalidate cache

    def A_inv(self) -> np.ndarray:
        if self._A_inv is None:
            self._A_inv = np.linalg.inv(self.A)
        return self._A_inv

    def theta(self) -> np.ndarray:
        return self.A_inv() @ self.b

    def ucb_score(self, phi: np.ndarray, alpha: float) -> float:
        th = self.theta()
        exploit = float(th @ phi)
        explore = alpha * float(np.sqrt(max(phi @ self.A_inv() @ phi, 0.0)))
        return exploit + explore


# ─────────────────────────────────────────────────────────────────────────────

class UCBPolicy(BasePolicy):
    name = "LinUCB"

    # [SỬA LỖI]: Tăng chiều lên 6 để chứa hệ số Bias
    FEATURE_DIM = 6

    def __init__(
        self,
        alpha: float = 1.0,
        deadline_ms: float = 500.0,    
        lam: float = 1.0,
        min_pulls: int = 20,            
    ):
        self.alpha = alpha
        self.deadline = deadline_ms / 1000.0
        self.lam = lam
        self.min_pulls = min_pulls

        self._arms: Dict[str | int, ArmState] = {}
        self._pull_count: Dict[str | int, int] = {}
        self._pending: Dict[int, tuple] = {}

    def decide(self, task: "Task", user_obs: dict, obs: dict) -> "str | int":
        self._init_arms_if_needed(obs)
        phi_map = self._build_context(task, user_obs, obs)

        under_explored = [
            arm for arm in self._arms
            if self._pull_count.get(arm, 0) < self.min_pulls
        ]

        if under_explored:
            # Chọn arm ít được thử nhất (không random hoàn toàn)
            best_arm = min(under_explored, key=lambda a: self._pull_count.get(a, 0))
        else:
            # UCB bình thường sau khi tất cả arm đã được warm-up
            best_arm = max(
                self._arms,
                key=lambda a: self._arms[a].ucb_score(phi_map[a], self.alpha),
            )

        self._pull_count[best_arm] = self._pull_count.get(best_arm, 0) + 1
        self._pending[task.task_id] = (best_arm, phi_map[best_arm])

        return best_arm

    def update(self, task: "Task"):
        if task.task_id not in self._pending:
            return
        if not task.done or np.isnan(task.latency):
            self._pending.pop(task.task_id, None)
            return

        arm_key, phi = self._pending.pop(task.task_id)
        # 1. Báo cáo Tốt (Xong DAG) và Xấu (Kẹt xe)
        is_last_task = (task.job_id is not None) and (len(task.successors) == 0)
        fallback_happened = getattr(task, 'is_fallback', False)
        
        # 1. Kiểm tra xem đây có phải là Task cuối cùng của DAG không
        is_last_task = (task.job_id is not None) and (len(task.successors) == 0)
        
        # 2. tính điểm Trễ
        reward_latency = 1.0 - (task.latency / self.deadline)
        
        # 3. Thêm Bonus nếu là task cuối DAG
        reward_bonus = 10.0 if is_last_task else 0.0
        
        reward_penalty = -10.0 if fallback_happened else 0.0
        # 4. reward tổng
        reward = reward_latency + reward_bonus + reward_penalty 

        if arm_key in self._arms:
            self._arms[arm_key].update(phi, reward)

    def _init_arms_if_needed(self, obs: dict):
        if self._arms:
            return
        self._arms["local"] = ArmState(self.FEATURE_DIM, self.lam)
        self._pull_count["local"] = 0
        for e in obs["edges"]:
            eid = e["edge_id"]
            self._arms[eid] = ArmState(self.FEATURE_DIM, self.lam)
            self._pull_count[eid] = 0

    def _build_context(
        self,
        task: "Task",
        user_obs: dict,
        obs: dict,
    ) -> Dict["str | int", np.ndarray]:
        
        local_load = user_obs.get("load", 0)
        cycles_norm = task.cycles / 1e9
        
        max_freq = max((e.get("cpu_freq", 1.0) for e in obs["edges"]), default=1.0)
        if max_freq == 0:
            max_freq = 1.0

        local_cpu_norm = cfg.USER_CPU_FREQ / max_freq
        phi_map: Dict[str | int, np.ndarray] = {}

        # [SỬA LỖI]: Thêm 1.0 vào cuối mảng
        phi_map["local"] = np.array([
            local_load / 10.0,
            0.0,
            local_cpu_norm,
            cycles_norm,
            0.0,
            1.0  # <--- HỆ SỐ BIAS
        ], dtype=float)

        DEFAULT_RATE = 20e6

        for e in obs["edges"]:
            eid = e["edge_id"]
            edge_queue = e.get("queue_len", e.get("total_load", 0))
            cpu_freq = e.get("cpu_freq", max_freq)
            rate = e.get("channel_rate", DEFAULT_RATE) or DEFAULT_RATE
            tx_est = task.input_bits / rate

            # [SỬA LỖI]: Thêm 1.0 vào cuối mảng
            phi_map[eid] = np.array([
                local_load / 10.0,
                edge_queue / 10.0,
                cpu_freq / max_freq,
                cycles_norm,
                min(tx_est, 1.0),
                1.0  # <--- HỆ SỐ BIAS
            ], dtype=float)

        return phi_map

    def __repr__(self) -> str:
        return f"UCBPolicy(alpha={self.alpha}, min_pulls={self.min_pulls})"