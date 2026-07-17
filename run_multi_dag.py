# File: run_single_dag.py
import os, random, math
import numpy as np
import config as cfg
from env.mec_env import MecEnv
from controller import Controller
from policy.heuristic_policy import HeuristicPolicy
from policy.ucb_policy import UCBPolicy
from policy.eps_greedy_policy import EpsGreedyPolicy
from env.trace_parser import DAGParser
import json

def main():
    os.makedirs("results", exist_ok=True)
    
    # 1. Các mức tải giống hệt Giai đoạn 1
    test_rates = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    DEADLINE_MS = 3000.0 # Nới lỏng deadline vì DAG thật tính toán rất nặng
    
    # =========================================================
    # BẠN MUỐN CHẠY APP NÀO THÌ BỎ COMMENT APP ĐÓ (CHỈ CHỌN 1)
    # =========================================================
    SELECTED_APP = "lightgbm"
    # SELECTED_APP = "mapreduce"
    # SELECTED_APP = "matrix"
    # SELECTED_APP = "video"
    
    app_to_json = {
        "lightgbm": "lightgbm.json",
        "mapreduce": "mapreduce.json",
        "matrix": "matrix_app.json",
        "video": "video_app.json"
    }
    # =========================================================

    policies_to_test = [
        ("AllLocal", HeuristicPolicy(mode="all_local")),
        ("AllEdge0", HeuristicPolicy(mode="all_edge", edge_id=0)),
        ("TypeAware", HeuristicPolicy(mode="type_aware")),
        ("EpsGreedy_Ctx", EpsGreedyPolicy(epsilon=0.1, deadline_ms=DEADLINE_MS)),
        ("LinUCB", UCBPolicy(alpha=1.0, deadline_ms=DEADLINE_MS))
    ]

    print("\n" + "="*98)
    print(f"| Đang chạy DAG: {SELECTED_APP.upper()} (Nhiễu = TẮT) |")
    print("="*98)
    print(f"| {'Rate':^6} | {'Policy':^20} | {'Tổng Task':^15} | {'Tổng DAG':^12} | {'Dropped':^9} | {'Độ Trễ TB (ms)':^16} |")
    print("-" * 98)

    parser = DAGParser()
    json_path = os.path.join("profile_data", app_to_json[SELECTED_APP])

    for rate in test_rates:
        for run_name, policy in policies_to_test:
            run_id = f"{SELECTED_APP}_{run_name}_Rate{rate}"
            
            random.seed(cfg.RANDOM_SEED)
            np.random.seed(cfg.RANDOM_SEED)
            
            # TẮT NHIỄU (enable_interference = False)
            env = MecEnv(seed=cfg.RANDOM_SEED, enable_interference=False)
            ctrl = Controller(policy=policy)
            
            env.reset()
            active_jobs = {}
            dag_tasks_to_schedule = []
            arrival_acc = 0.0
            next_job_id = 1
            prev = 0

            parser.current_task_id = 0 
            total_tasks_generated = 0 # Biến đếm tổng task thật

            while not env.done:
                # Dừng sinh task ở 60s như code cũ của bạn
                if env.sim_time < getattr(cfg, 'TASK_GEN_DURATION', 60.0):
                    arrival_acc += rate * cfg.DT
                else:
                    arrival_acc = 0  
                
                n_arrivals = int(arrival_acc)
                arrival_acc -= n_arrivals
                
                tasks_ready_now = []
                for _ in range(n_arrivals):
                    u_id = random.randint(0, cfg.NUM_USERS - 1)
                    
                    if os.path.exists(json_path):
                        new_job = parser.parse_job(json_path, next_job_id, u_id, env.sim_time, SELECTED_APP)
                        active_jobs[new_job.job_id] = new_job
                        next_job_id += 1
                        total_tasks_generated += len(new_job.tasks) # Cộng dồn số task con
                        tasks_ready_now.extend(t for t in new_job.tasks.values() if t.ready_to_start)

                if dag_tasks_to_schedule:
                    tasks_ready_now.extend(dag_tasks_to_schedule)
                    dag_tasks_to_schedule = []

                ctrl.step(env, tasks_ready_now)

                new_done = env.finished_tasks[prev:]
                prev = len(env.finished_tasks)
                
                for task in new_done:
                    if hasattr(policy, 'update'):
                        policy.update(task)

                    if getattr(task, "job_id", None) is not None:
                        job = active_jobs.get(task.job_id)
                        if job and not job.is_completed:
                            unlocked_tasks = job.update_task_completion(task.task_id, env.sim_time)
                            dag_tasks_to_schedule.extend(unlocked_tasks)
                            
                # DỪNG SỚM NẾU XONG HẾT
                if env.sim_time >= getattr(cfg, 'TASK_GEN_DURATION', 60.0):
                    if all(j.is_completed for j in active_jobs.values()):
                        break

            # --- TỔNG KẾT ---
            completed_dags = [j for j in active_jobs.values() if j.is_completed]
            total_dags = len(completed_dags)
            avg_dag_lat = (sum(j.latency for j in completed_dags) / total_dags * 1000) if total_dags > 0 else 0
            
            # Tính Tỉ lệ Task
            total_tasks_done = len(env.finished_tasks)
            task_ratio_str = f"{total_tasks_done} / {total_tasks_generated}"
            
            # Tính Tỉ lệ DAG
            dag_ratio_str = f"{total_dags} / {len(active_jobs)}"
            
            # TÍNH DROPPED (y hệt code cũ)
            dropped_count = len(getattr(env, 'dropped_tasks', []))
            
            print(f"| {rate:^6.2f} | {run_name:<20} | {task_ratio_str:^15} | {dag_ratio_str:^12} | {dropped_count:^9} | {avg_dag_lat:>12.2f} ms |")
            
            # LƯU FILE JSON GỐC
            summary_data = {
                "total_done": total_dags,
                "total_generated": len(active_jobs),
                "latency_all_ms": {"mean": avg_dag_lat}
            }
            json_filename = f"results/RealDAG_{SELECTED_APP}_{run_name}_Rate{rate}_summary.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(summary_data, f)
                
        if rate != test_rates[-1]:
            print("-" * 98)
            
    print("="*98)

if __name__ == "__main__":
    main()