# File: policy/heuristic_policy.py

class HeuristicPolicy:
    def __init__(self, mode="all_local", edge_id=0):
        self.mode = mode
        self.edge_id = edge_id
        self._pending = {}
        self.deadline = 500.0 

    def act(self, tasks, obs):
        actions = []
        for t in tasks:
            if self.mode == "all_local":
                action = "local"
            elif self.mode == "all_edge":
                action = self.edge_id
            elif self.mode == "type_aware":
                if t.task_type == "Light":
                    action = 0       # Light đẩy lên Edge 0
                elif t.task_type == "Medium":
                    action = 1       # Medium đẩy lên Edge 1
                else:
                    action = "local" # Heavy bắt buộc ở lại Local
            else:
                action = "local"
                
            actions.append((t, action))
            
            sim_time = obs['sim_time'] if isinstance(obs, dict) and 'sim_time' in obs else 0
            self._pending[t.task_id] = (action, sim_time)
            
        return actions

    def update(self, completed_task):
        pass

    def __repr__(self):
        return f"Heuristic({self.mode})"