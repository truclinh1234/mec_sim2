# File: main_real_dag.py
import os, random, math
import numpy as np
import config as cfg
from env.mec_env import MecEnv
from controller import Controller
from policy.heuristic_policy import HeuristicPolicy
from policy.ucb_policy import UCBPolicy
from policy.eps_greedy_policy import EpsGreedyPolicy
from env.trace_parser import DAGParser
from metrics.collector import MetricsCollector

def main():
    os.makedirs("results", exist_ok=True)
    
    # Đồ thị thực tế LightGBM rất lớn, nên ta phải hạ Rate xuống cực thấp 
    # (VD: Rate = 0.2 nghĩa là 5 giây mới sinh ra 1 DAG)
    test_rates = [2, 4, 6, 8, 10] 
    
    DEADLINE_MS = 1000.0 # Nới lỏng deadline vì DAG thật rất nặng
    
    policies_to_test = [
        ("AllLocal", HeuristicPolicy(mode="all_local")),
        ("AllEdge0", HeuristicPolicy(mode="all_edge", edge_id=0)),
        ("HalfHalf", HeuristicPolicy(mode="half_half", edge_id=0)), # Thuật toán thông minh của bạn
        ("UCB_Alpha_1.0", UCBPolicy(alpha=1.0, deadline_ms=DEADLINE_MS)),
        ("EpsGreedy_0.1", EpsGreedyPolicy(epsilon=0.1, deadline_ms=DEADLINE_MS))
    ]

    print("\n" + "="*70)
    print(f"| {'Rate':^6} | {'Chính sách (Policy)':^20} | {'Tổng DAG Xong':^16} | {'Độ Trễ TB (ms)':^16} |")
    print("-" * 70)

    # Khởi tạo Parser đọc file JSON
    parser = DAGParser()
    json_path = os.path.join("profile_data", "lightgbm.json")

    for rate in test_rates:
        for run_name, policy in policies_to_test:
            run_id = f"RealDAG_{run_name}_Rate{rate}"
            
            random.seed(cfg.RANDOM_SEED)
            np.random.seed(cfg.RANDOM_SEED)
            
            env = MecEnv(seed=cfg.RANDOM_SEED, enable_interference=False)
            ctrl = Controller(policy=policy)
            col = MetricsCollector(run_name=run_id)
            
            env.reset()
            active_jobs = {}
            dag_tasks_to_schedule = []
            arrival_acc = 0.0
            next_job_id = 1
            prev = 0

            # Reset lại ID của parser cho mỗi lượt chạy
            parser.current_task_id = 0 

            while not env.done:
                arrival_acc += rate * cfg.DT
                n_arrivals = int(arrival_acc)
                arrival_acc -= n_arrivals
                
                tasks_ready_now = []
                for _ in range(n_arrivals):
                    u_id = random.randint(0, cfg.NUM_USERS - 1)
                    
                    # ĐỌC TRỰC TIẾP TỪ FILE JSON
                    if os.path.exists(json_path):
                        new_job = parser.parse_job(json_path, next_job_id, u_id, env.sim_time, "lightgbm")
                        active_jobs[new_job.job_id] = new_job
                        next_job_id += 1
                        tasks_ready_now.extend(t for t in new_job.tasks.values() if t.ready_to_start)

                if dag_tasks_to_schedule:
                    tasks_ready_now.extend(dag_tasks_to_schedule)
                    dag_tasks_to_schedule = []

                obs = env.get_obs()
                actions = policy.act(tasks_ready_now, obs)
                env.apply_actions(actions)
                env.step()

                new_done = env.finished_tasks[prev:]
                prev = len(env.finished_tasks)
                
                for task in new_done:
                    arm_key = None
                    if hasattr(policy, '_pending') and task.task_id in policy._pending:
                        arm_key = policy._pending[task.task_id][0]

                    if hasattr(policy, 'update'):
                        policy.update(task)

                    if task.done and not math.isnan(task.latency):
                        reward = max(-1.0, 1.0 - task.latency / (DEADLINE_MS/1000.0))
                        col.register_ucb_reward(task.task_id, reward)
                        if arm_key is not None:
                            display_arm = "Local" if arm_key == "local" else f"Edge {arm_key}"
                            col.register_ucb_arm(task.task_id, display_arm)

                    if getattr(task, "job_id", None) is not None:
                        job = active_jobs.get(task.job_id)
                        if job and not job.is_completed:
                            unlocked_tasks = job.update_task_completion(task.task_id, env.sim_time)
                            dag_tasks_to_schedule.extend(unlocked_tasks)
                            
                col.on_tasks_done(new_done)
                col.tick(env.sim_time, obs)

            completed_dags = [j for j in active_jobs.values() if j.is_completed]
            total_dags = len(completed_dags)
            avg_dag_lat = (sum(j.latency for j in completed_dags) / total_dags * 1000) if total_dags > 0 else 0
            
            # Tạo chuỗi tỷ lệ: Số DAG hoàn thành / Tổng số DAG sinh ra
            dag_ratio_str = f"{total_dags} / {len(active_jobs)}"
            
            summary = env.summary()
            summary["policy"] = repr(policy)
            col.save_all(summary)
            
            # In kết quả theo định dạng tỷ lệ mới
            print(f"| {rate:^6} | {run_name:<20} | {dag_ratio_str:^16} | {avg_dag_lat:>12.2f} ms  |")

    print("="*70)

if __name__ == "__main__":
    main()