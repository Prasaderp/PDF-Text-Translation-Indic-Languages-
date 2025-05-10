import torch
import time
import config
from model_loader import initialize_model, set_model_tokenizer

def reset_gpu_memory():
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("Refreshing GPU memory...")
        start_time = time.time()
        
        new_tokenizer, new_model = initialize_model()
        
        print(f"GPU memory refreshed in {time.time()-start_time:.2f}s")

def check_memory_and_reset(total_pages):
    if config.DEVICE != "cuda" or total_pages <= 5:
        return False
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated()
    if allocated_memory / total_memory > config.MEMORY_THRESHOLD:
        reset_gpu_memory()
        return True
    return False
