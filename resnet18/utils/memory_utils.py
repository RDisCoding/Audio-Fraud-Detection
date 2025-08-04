import gc
import psutil
import os
import torch
import numpy as np
from pathlib import Path

class MemoryManager:
    """Utility class for memory management during preprocessing and training"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        memory_info = self.process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    
    def get_memory_percent(self):
        """Get current memory usage as percentage of total system memory"""
        return self.process.memory_percent()
    
    def clear_cache(self):
        """Clear various caches to free memory"""
        # Clear Python garbage collection
        gc.collect()
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def force_garbage_collection(self):
        """Force aggressive garbage collection"""
        for _ in range(3):
            gc.collect()
    
    def monitor_memory(self, func, *args, **kwargs):
        """Monitor memory usage during function execution"""
        initial_memory = self.get_memory_usage()
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        
        result = func(*args, **kwargs)
        
        final_memory = self.get_memory_usage()
        memory_diff = final_memory - initial_memory
        
        print(f"Final memory usage: {final_memory:.2f} MB")
        print(f"Memory difference: {memory_diff:+.2f} MB")
        
        return result
    
    def check_memory_limit(self, limit_percent=85):
        """Check if memory usage exceeds limit"""
        current_percent = self.get_memory_percent()
        if current_percent > limit_percent:
            print(f"Warning: Memory usage at {current_percent:.1f}% (limit: {limit_percent}%)")
            self.clear_cache()
            return True
        return False
    
    def safe_array_operation(self, operation_func, *args, **kwargs):
        """Safely perform array operations with memory monitoring"""
        try:
            self.check_memory_limit()
            result = operation_func(*args, **kwargs)
            self.clear_cache()
            return result
        except MemoryError:
            print("Memory error occurred, clearing cache and retrying...")
            self.clear_cache()
            # Optionally retry with smaller batch size or different approach
            raise

def batch_process_with_memory_check(data, batch_size, process_func, memory_limit=85):
    """Process data in batches with memory monitoring"""
    memory_manager = MemoryManager()
    results = []
    
    for i in range(0, len(data), batch_size):
        # Check memory before processing each batch
        if memory_manager.check_memory_limit(memory_limit):
            print(f"Memory limit reached, processing smaller batch")
            # Reduce batch size if memory is too high
            current_batch_size = max(1, batch_size // 2)
        else:
            current_batch_size = batch_size
        
        batch = data[i:i + current_batch_size]
        
        try:
            batch_result = process_func(batch)
            results.extend(batch_result)
            
            # Clear memory after each batch
            memory_manager.clear_cache()
            
        except MemoryError:
            print(f"Memory error at batch {i//batch_size}, reducing batch size")
            # Try with smaller batch size
            smaller_batch_size = max(1, current_batch_size // 2)
            for j in range(0, len(batch), smaller_batch_size):
                small_batch = batch[j:j + smaller_batch_size]
                small_result = process_func(small_batch)
                results.extend(small_result)
                memory_manager.clear_cache()
    
    return results

def optimize_numpy_memory():
    """Set numpy to use less memory"""
    # Disable numpy multi-threading to reduce memory overhead
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

def get_optimal_batch_size(sample_size_mb, available_memory_gb, safety_factor=0.5):
    """Calculate optimal batch size based on available memory"""
    available_memory_mb = available_memory_gb * 1024 * safety_factor
    optimal_batch_size = max(1, int(available_memory_mb / sample_size_mb))
    return optimal_batch_size

def cleanup_variables(*variables):
    """Clean up specific variables and force garbage collection"""
    for var in variables:
        if var is not None:
            del var
    gc.collect()
