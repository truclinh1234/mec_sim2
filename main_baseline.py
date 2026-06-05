import os, random, math
import numpy as np
import config as cfg
from env.mec_env import MecEnv
from env.task import Task
from controller import Controller
from policy.heuristic_policy import HeuristicPolicy
from metrics.collector import MetricsCollector

def main():
    os.makedirs("results", exist_ok=True)
    
    # Đổi rate thành các con số thực tế hơn để hệ thống không bị "chết ngộp"
    test_rates = [1.2, 1.4, 1.6, 1.8] 
    
    policies_to_test = [
        ("AllLocal", HeuristicPolicy(mode="all_local")),
        ("AllEdge0", HeuristicPolicy(mode="all_edge", edge_id=0))
    ]

    # Danh sách để lưu trữ kết quả cuối cùng in ra bảng
    table_results = []

    for rate in test_rates:
        for run_name, policy in policies_to_test:
            run_id = f"Baseline_{run_name}_Rate{rate}"
            print(f"⏳ Đang chạy: {run_id} ...")

            random.seed(cfg.RANDOM_SEED)
            np.random.seed(cfg.RANDOM_SEED)
            
            env = MecEnv(seed=cfg.RANDOM_SEED, enable_interference=False)
            ctrl = Controller(policy=policy)
            col = MetricsCollector(run_name=run_id)

            env.reset()
            prev = 0
            arrival_acc = 0.0

            while not env.done:
                arrival_acc += rate * cfg.DT
                n_arrivals = int(arrival_acc)
                arrival_acc -= n_arrivals
                
                tasks = []
                for _ in range(n_arrivals):
                    u_id = random.randint(0, cfg.NUM_USERS - 1)
                    user = env.users[u_id]
                    
                    task_type_dict = {"name": "Heavy", "cycles": 2e9, "input_bits": 20e6}
                    gap = env.sim_time - user._last_arrival_time if user._last_arrival_time is not None else None
                    user._last_arrival_time = env.sim_time
                    
                    task = Task(
                        task_id           = user._next_id(),
                        user_id           = u_id,
                        task_type         = task_type_dict["name"],
                        cycles            = task_type_dict["cycles"],
                        input_bits        = task_type_dict["input_bits"],
                        arrival_time      = env.sim_time,
                        inter_arrival_gap = gap,
                    )
                    tasks.append(task)
                
                obs = env.get_obs()
                actions = policy.act(tasks, obs)
                env.apply_actions(actions)
                env.step()

                new_done = env.finished_tasks[prev:]
                prev = len(env.finished_tasks)
                for task in new_done:
                    if task.done and not math.isnan(task.latency):
                        col.register_ucb_reward(task.task_id, 0)
                
                col.on_tasks_done(new_done)
                col.tick(env.sim_time, obs)

            summary = env.summary()
            summary["policy"] = repr(policy)
            col.save_all(summary)

            lat = summary.get("latency_all_ms", {})
            mean_lat = lat.get('mean', '—')
            
            # Lưu lại dữ liệu để in bảng
            table_results.append({
                "rate": rate,
                "policy": run_name,
                # Thay đổi ở đây: kết hợp Số task xong / Tổng số task sinh ra thành một chuỗi
                "tasks_done": f"{summary['total_done']} / {summary['total_generated']}",
                "mean_lat": mean_lat
            })

    # ==========================================
    # IN KẾT QUẢ DƯỚI DẠNG BẢNG
    # ==========================================
    print("\n" + "="*70)
    print(f"| {'Rate':^6} | {'Chính sách (Policy)':^20} | {'Tổng Task Xong':^16} | {'Độ Trễ TB (ms)':^16} |")
    print("-" * 70)
    
    for res in table_results:
        # Nếu là AllLocal, in thêm vạch ngăn cách cho dễ nhìn
        if res['policy'] == "AllLocal" and res['rate'] != test_rates[0]:
            print("-" * 70)
            
        print(f"| {res['rate']:^6} | {res['policy']:<20} | {res['tasks_done']:^16} | {str(res['mean_lat']):>12} ms  |")
    
    print("="*70)
    print("\n→ Hoàn thành! Kết quả chi tiết đã được lưu ở folder results/")

if __name__ == "__main__":
    main()