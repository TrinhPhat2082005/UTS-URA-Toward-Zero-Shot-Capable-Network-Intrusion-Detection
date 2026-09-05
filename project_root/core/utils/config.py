import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "./project/data/processed_csv")
IMAGE_DIR = os.path.join(BASE_DIR, "./data/vlm_dataset")
FUN_DIR = os.path.join(BASE_DIR, "./project_root/core/utils")
WINDOW_DURATION = 10 
FEATURES = [
    'Flow Pkts/s', 'Flow IAT Std', 'Flow Duration', 
    'Fwd Pkt Len Mean', 'Pkt Len Std', 
    'SYN Flag Cnt', 'ACK Flag Cnt', 'RST Flag Cnt'
]
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" 
