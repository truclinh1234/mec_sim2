import os
import numpy as np
import config as cfg
from env.mec_env import MecEnv
from env.task import Task
from env.dag_job import DAGJob

# Lấy cấu hình chuẩn của 3 loại Task từ config.py
TASK_CONFIGS = {t["name"]: t for t in cfg.TASK_TYPES}

def create_custom_linear_dag(job_id, user_id, arrival_time, num_nodes, task_type_name):
    job = DAGJob(job_id=job_id, app_type=f"Linear_{task_type_name}", arrival_time=arrival_time)
    
    cfg_cycles = TASK_CONFIGS[task_type_name]["cycles"]
    cfg_bits = TASK_CONFIGS[task_type_name]["input_bits"]
    
    # 1. Khởi tạo tất cả các Node
    for i in range(num_nodes):
        task_id = job_id * 1000 + i  # Đánh id riêng biệt
        t = Task(
            task_id=task_id, 
            user_id=user_id,
            task_type=task_type_name, 
            cycles=cfg_cycles, 
            input_bits=cfg_bits, 
            arrival_time=arrival_time
        )
        t.job_id = job_id
        t.dag_name = f"Node_{i}"
        t.app_type = f"Linear_{task_type_name}"
        job.add_task(t)
        
    # 2. Xâu chuỗi thành Linear: 0 -> 1 -> 2 -> ... -> (N-1)
    for i in range(num_nodes - 1):
        job.add_dependency(job_id * 1000 + i, job_id * 1000 + i + 1)
        
    # 3. Chỉ Node 0 được phép chạy ngay lập tức
    for i in range(num_nodes):
        job.tasks[job_id * 1000 + i].ready_to_start = (i == 0)
        
    return job


def run_linearity_experiment():
    print("\n" + "="*85)
    print("THÍ NGHIỆM ĐÁNH GIÁ TÍNH TUYẾN TÍNH CỦA 3 LOẠI DAG (LIGHT, MEDIUM, HEAVY)")
    print("="*85)
    
    # Test các mốc số lượng Node để thấy rõ độ dốc
    node_counts = [1, 3, 5, 10, 20]
    task_types = ["Light", "Medium", "Heavy"]
    
    for t_type in task_types:
        print(f"\n--- ĐANG TEST CHUỖI RÒNG: {t_type.upper()} ---")
        print(f"| {'Số Node (N)':^12} | {'Độ Trễ Thực Tế đo được':^25} | {'Độ Trễ Lý Thuyết (L = N*C)':^26} |")
        print("-" * 71)
        
        # Tính trễ lý thuyết 1 task (Máy Local CPU 1GHz)
        base_cycles = TASK_CONFIGS[t_type]["cycles"]
        c_theory_ms = (base_cycles / 1e9) * 1000.0 
        
        for N in node_counts:
            # Môi trường siêu sạch, không nhiễu
            env = MecEnv(enable_interference=False)
            env.reset()
            
            # Tạo DAG đồng nhất
            job = create_custom_linear_dag(job_id=1, user_id=0, arrival_time=0.0, num_nodes=N, task_type_name=t_type)
            dag_tasks_to_schedule = [t for t in job.tasks.values() if t.ready_to_start]
            
            while not env.done and not job.is_completed:
                # Ép chạy tất cả trên máy Local để đo sức mạnh CPU thuần túy
                actions = [(t, "local") for t in dag_tasks_to_schedule]
                env.apply_actions(actions)
                dag_tasks_to_schedule = []
                
                prev_done = len(env.finished_tasks)
                env.step()
                
                for task in env.finished_tasks[prev_done:]:
                    dag_tasks_to_schedule.extend(job.update_task_completion(task.task_id, env.sim_time))
                    
                if job.is_completed:
                    break
                    
            actual_latency_ms = job.latency * 1000
            theoretical_latency_ms = N * c_theory_ms 
            
            print(f"| {N:^12} | {actual_latency_ms:>22.2f} ms | {theoretical_latency_ms:>23.2f} ms |")

if __name__ == "__main__":
    run_linearity_experiment()