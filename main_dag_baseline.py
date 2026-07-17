# File: main_dag_baseline.py
import os, random
import numpy as np
import config as cfg
from env.mec_env import MecEnv
from controller import Controller
from policy.heuristic_policy import HeuristicPolicy
from env.simple_dag import create_linear_dag
from policy.eps_greedy_policy import EpsGreedyPolicy
from policy.standard_eps_greedy import StandardEpsGreedyPolicy
from policy.ucb_policy import UCBPolicy

def main():
    os.makedirs("results", exist_ok=True)
    
    # Ta test ở Rate = 1 (rất thong thả) để đo độ trễ chuẩn của 1 DAG
    test_rates = [0.5, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 20.0]
   
    print("\n" + "="*98)
    print(f"| {'Rate':^6} | {'Policy':^20} | {'Tổng Task':^15} | {'Tổng DAG':^12} | {'Dropped':^9} | {'Độ Trễ TB (ms)':^16} |")
    print("-" * 98)

    for rate in test_rates:
        policies_to_test = [
        ("AllLocal", HeuristicPolicy(mode="all_local")),
        ("AllEdge0", HeuristicPolicy(mode="all_edge", edge_id=0)),
        ("TypeAware", HeuristicPolicy(mode="type_aware")),   
        # ("MAB_NoContext", StandardEpsGreedyPolicy(epsilon=0.1)),
        ("EpsGreedy_Ctx", EpsGreedyPolicy(epsilon=0.1, deadline_ms=3000.0)),
        ("LinUCB", UCBPolicy(alpha=0.1, deadline_ms=3000.0, min_pulls=10))
    ]
        for run_name, policy in policies_to_test:
            run_id = f"DAG_{run_name}_Rate{rate}"
            
            random.seed(cfg.RANDOM_SEED)
            np.random.seed(cfg.RANDOM_SEED)
            
            env = MecEnv(seed=cfg.RANDOM_SEED, enable_interference=True)
            ctrl = Controller(policy=policy)
            
            env.reset()
            active_jobs = {}
            dag_tasks_to_schedule = []
            arrival_acc = 0.0
            next_job_id = 1

            while not env.done:
                # 1. Sinh DAGJob mới — CHỈ TRONG 60 GIÂY ĐẦU
                if env.sim_time < getattr(cfg, 'TASK_GEN_DURATION', 60.0):
                    arrival_acc += rate * cfg.DT
                else:
                    arrival_acc = 0  # Ngừng ném task, chỉ đợi xử lý nốt hàng đợi
                    
                n_arrivals = int(arrival_acc)
                arrival_acc -= n_arrivals
                
                tasks_ready_now = []
                for _ in range(n_arrivals):
                    u_id = random.randint(0, cfg.NUM_USERS - 1)
                    new_job = create_linear_dag(next_job_id, u_id, env.sim_time)
                    active_jobs[new_job.job_id] = new_job
                    next_job_id += 1
                    # Góp các Node_0 vào danh sách sẵn sàng chạy
                    tasks_ready_now.extend(t for t in new_job.tasks.values() if t.ready_to_start)

                if dag_tasks_to_schedule:
                    tasks_ready_now.extend(dag_tasks_to_schedule)
                    dag_tasks_to_schedule = []

                # 2. Xử lý logic môi trường
                # Dùng Controller để xử lý cả 2 loại policy
                ctrl.step(env, tasks_ready_now)
                
                prev_done_count = len(env.finished_tasks)
                env.step()

                # 3. Mở khóa các Node tiếp theo (Dependency)
                new_done = env.finished_tasks[prev_done_count:]
                for task in new_done:
                    if hasattr(policy, 'update'):
                        policy.update(task)
                    if getattr(task, "job_id", None) is not None:
                        job = active_jobs.get(task.job_id)
                        if job and not job.is_completed:
                            unlocked_tasks = job.update_task_completion(task.task_id, env.sim_time)
                            dag_tasks_to_schedule.extend(unlocked_tasks)
                # --- DỪNG SỚM NẾU TẤT CẢ ĐÃ XONG ---
                if env.sim_time >= getattr(cfg, 'TASK_GEN_DURATION', 60.0):
                    if all(j.is_completed for j in active_jobs.values()):
                        break

                        # --- TỔNG KẾT VÀ TÍNH ĐỘ TRỄ ---
            completed_dags = [j for j in active_jobs.values() if j.is_completed]
            total_dags = len(completed_dags)
            avg_dag_lat = (sum(j.latency for j in completed_dags) / total_dags * 1000) if total_dags > 0 else 0
            
            # --- TÍNH TỈ LỆ TASK ---
            total_tasks_done = len(env.finished_tasks)
            total_tasks_generated = len(active_jobs) * 3
            task_ratio_str = f"{total_tasks_done} / {total_tasks_generated}"
            
            # --- TÍNH TỈ LỆ DAG ---
            dag_ratio_str = f"{total_dags} / {len(active_jobs)}"
            
            # In dòng kết quả (chỉ thêm .2f vào biến rate)
            # --- TÍNH SỐ LƯỢNG DROP ---
            dropped_count = len(getattr(env, 'dropped_tasks', []))
            
            print(f"| {rate:^6.2f} | {run_name:<20} | {task_ratio_str:^15} | {dag_ratio_str:^12} | {dropped_count:^9} | {avg_dag_lat:>12.2f} ms |")
            import json
            summary_data = {
                "total_done": total_dags,
                "total_generated": len(active_jobs),
                "latency_all_ms": {"mean": avg_dag_lat}
            }
            json_filename = f"results/Baseline_{run_name}_Rate{rate}_summary.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(summary_data, f)
        if rate != test_rates[-1]:
            print("-" * 98)
    print("="*98)

if __name__ == "__main__":
    main()