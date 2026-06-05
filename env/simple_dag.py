# File: env/simple_dag.py
from env.dag_job import DAGJob
from env.task import Task

def create_linear_dag(job_id: int, user_id: int, arrival_time: float) -> DAGJob:
    """
    Tạo 1 DAG nối tiếp nhau. Số lượng Node tự động thích ứng với mảng task_configs.
    """
    job = DAGJob(job_id=job_id, app_type="SimpleLinear", arrival_time=arrival_time)
    
    # =============================================================
    # BẠN HÃY UNCOMMENT 1 TRONG 4 KHỐI DƯỚI ĐÂY ĐỂ CHẠY TEST TƯƠNG ỨNG
    # =============================================================

    # Khối 0: DAG HỖN HỢP (Dùng để chứng minh sức mạnh của TypeAware Policy)
    task_configs = [
        # {"name": "Light",  "cycles": 3e8, "input_bits": 1e6},
        # {"name": "Medium", "cycles": 1e9, "input_bits": 5e6},
        # {"name": "Heavy",  "cycles": 2e9, "input_bits": 20e6}
    ]

    # Khối 1: DAG toàn Light (Bỏ comment khi muốn chạy Ablation Study)
    task_configs = [
        {"name": "Light", "cycles": 3e8, "input_bits": 1e6},
        {"name": "Light", "cycles": 3e8, "input_bits": 1e6},
        {"name": "Light", "cycles": 3e8, "input_bits": 1e6}
    ]

    # Khối 2: DAG toàn Medium (Bỏ comment khi muốn chạy Ablation Study)
    # task_configs = [
    #     {"name": "Medium", "cycles": 1e9, "input_bits": 5e6},
    #     {"name": "Medium", "cycles": 1e9, "input_bits": 5e6},
    #     {"name": "Medium", "cycles": 1e9, "input_bits": 5e6}
    # ]

    # Khối 3: DAG toàn Heavy (Bỏ comment khi muốn chạy Ablation Study)
    # task_configs = [
    #     {"name": "Heavy", "cycles": 2e9, "input_bits": 20e6},
    #     {"name": "Heavy", "cycles": 2e9, "input_bits": 20e6},
    #     {"name": "Heavy", "cycles": 2e9, "input_bits": 20e6}
    # ]
    
    # 1. Khởi tạo tasks
    for i, cfg in enumerate(task_configs):
        t = Task(
            task_id = job_id * 100 + i, 
            user_id = user_id,
            task_type = cfg["name"],
            cycles = cfg["cycles"],       
            input_bits = cfg["input_bits"],   
            arrival_time = arrival_time
        )
        t.job_id = job_id
        t.dag_name = f"Node_{i}"
        t.app_type = "SimpleLinear"
        job.add_task(t)
        
    # --- ĐOẠN NÀY ĐÃ ĐƯỢC LÀM CHO TỰ ĐỘNG ---
    num_nodes = len(task_configs)
    
    # 2. Xâu chuỗi thành Linear tự động: có bao nhiêu Node thì nối bấy nhiêu vòng xích
    for i in range(num_nodes - 1):
        job.add_dependency(job_id * 100 + i, job_id * 100 + i + 1)
    
    # 3. Chỉ Node_0 được phép chạy ngay lập tức, các Node khác tự động khóa
    for i in range(num_nodes):
        job.tasks[job_id * 100 + i].ready_to_start = (i == 0)
    
    return job