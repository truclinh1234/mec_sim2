# File: policy/standard_eps_greedy.py
import random
import numpy as np

class StandardEpsGreedyPolicy:
    """
    Multi-Armed Bandit cơ bản (KHÔNG CÓ CONTEXT).
    Chỉ duy trì giá trị kỳ vọng (Q-value) trung bình cho mỗi Arm một cách mù quáng.
    """
    def __init__(self, epsilon=0.1, deadline_ms=500.0, edge_ids=None):
        self.epsilon = epsilon
        self.deadline = deadline_ms / 1000.0
        
        if edge_ids is None:
            edge_ids = [0, 1] # Giả định ban đầu có Edge 0 và Edge 1
            
        self.arms = ["local"] + edge_ids
        self.Q = {arm: 0.0 for arm in self.arms}
        self.N = {arm: 0 for arm in self.arms}
        self._pending = {}

    def decide(self, task, user_obs, obs):
        # 1. Cập nhật danh sách Arm nếu có Edge mới xuất hiện
        for e in obs.get("edges", []):
            eid = e["edge_id"]
            if eid not in self.arms:
                self.arms.append(eid)
                self.Q[eid] = 0.0
                self.N[eid] = 0

        # 2. Epsilon-Greedy: Khám phá hay Khai thác?
        if random.random() < self.epsilon:
            chosen_arm = random.choice(self.arms)
        else:
            # Chọn Arm có Q-value cao nhất hiện tại
            max_q = max(self.Q.values())
            best_arms = [arm for arm, q in self.Q.items() if q == max_q]
            chosen_arm = random.choice(best_arms)
            
        self._pending[task.task_id] = chosen_arm
        return chosen_arm

    def update(self, task):
        if task.task_id not in self._pending:
            return
        if not task.done or np.isnan(task.latency):
            self._pending.pop(task.task_id, None)
            return

        chosen_arm = self._pending.pop(task.task_id)
        
        # Tính Reward: Trễ càng thấp, reward càng cao.
        
       
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

        


        
        
        # Cập nhật Q-value theo trung bình cộng (Incremental Mean)
        self.N[chosen_arm] += 1
        alpha = 1.0 / self.N[chosen_arm]
        self.Q[chosen_arm] += alpha * (reward - self.Q[chosen_arm])

    def __repr__(self):
        return f"MAB_NoContext(eps={self.epsilon})"