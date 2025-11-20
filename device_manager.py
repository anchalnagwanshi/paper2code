"""
device_manager.py - Intelligent GPU/CPU detection and management
"""

import torch
import logging
import platform
from typing import Dict, Optional, Tuple
import subprocess

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manage CPU/GPU device selection and optimization."""
    
    def __init__(self):
        self.device = None
        self.device_name = None
        self.gpu_available = False
        self.cuda_version = None
        self.device_info = {}
        
        self._detect_devices()
    
    def _detect_devices(self):
        """Detect available devices and their capabilities."""
        
        logger.info("="*60)
        logger.info("DEVICE DETECTION")
        logger.info("="*60)
        
        # Check CUDA availability
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            try:
                self.device = torch.device("cuda")
                self.device_name = torch.cuda.get_device_name(0)
                self.cuda_version = torch.version.cuda
                
                # Get detailed GPU info
                self.device_info = {
                    "type": "GPU",
                    "name": self.device_name,
                    "cuda_version": self.cuda_version,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
                    "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB",
                    "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
                }
                
                logger.info(f"✅ GPU Available: {self.device_name}")
                logger.info(f"   CUDA Version: {self.cuda_version}")
                logger.info(f"   GPU Memory: {self.device_info['memory_total']}")
                
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}")
                self._fallback_to_cpu()
        else:
            self._fallback_to_cpu()
        
        logger.info("="*60)
    
    def _fallback_to_cpu(self):
        """Fallback to CPU when GPU is not available."""
        self.device = torch.device("cpu")
        self.device_name = platform.processor() or "CPU"
        self.gpu_available = False
        
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        
        self.device_info = {
            "type": "CPU",
            "name": self.device_name,
            "physical_cores": cpu_count,
            "logical_cores": cpu_count_logical,
            "memory_total": f"{memory.total / 1e9:.2f} GB",
            "memory_available": f"{memory.available / 1e9:.2f} GB"
        }
        
        logger.info(f"⚠️  No GPU available, using CPU")
        logger.info(f"   Processor: {self.device_name}")
        logger.info(f"   Cores: {cpu_count} physical, {cpu_count_logical} logical")
        logger.info(f"   Memory: {self.device_info['memory_available']} / {self.device_info['memory_total']}")
    
    def get_device(self) -> torch.device:
        """Get the selected device."""
        return self.device
    
    def get_device_info(self) -> Dict:
        """Get detailed device information."""
        return self.device_info
    
    def optimize_batch_size(self, default_batch_size: int, model_type: str = "bert") -> int:
        """
        Automatically adjust batch size based on available device.
        
        Args:
            default_batch_size: The requested batch size
            model_type: Type of model (bert, resnet, etc.)
            
        Returns:
            Optimized batch size
        """
        
        if self.gpu_available:
            # GPU available - can use larger batches
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                # Adjust based on GPU memory and model type
                if model_type.lower() in ["bert", "gpt", "transformer"]:
                    # Transformers are memory-hungry
                    if gpu_memory < 6:
                        recommended = min(default_batch_size, 16)
                    elif gpu_memory < 12:
                        recommended = min(default_batch_size, 32)
                    else:
                        recommended = default_batch_size
                else:
                    # CNNs are less memory-hungry
                    if gpu_memory < 6:
                        recommended = min(default_batch_size, 32)
                    elif gpu_memory < 12:
                        recommended = min(default_batch_size, 64)
                    else:
                        recommended = default_batch_size
                
                if recommended != default_batch_size:
                    logger.warning(
                        f"Adjusting batch size from {default_batch_size} to {recommended} "
                        f"based on GPU memory ({gpu_memory:.1f} GB)"
                    )
                
                return recommended
                
            except Exception as e:
                logger.warning(f"Could not optimize batch size: {e}")
                return default_batch_size
        else:
            # CPU only - use smaller batches
            if model_type.lower() in ["bert", "gpt", "transformer"]:
                recommended = min(default_batch_size, 8)  # Very small for transformers on CPU
            else:
                recommended = min(default_batch_size, 32)
            
            if recommended != default_batch_size:
                logger.warning(
                    f"CPU detected: Reducing batch size from {default_batch_size} to {recommended} "
                    f"for better performance"
                )
            
            return recommended
    
    def optimize_num_workers(self) -> int:
        """Get optimal number of DataLoader workers."""
        if self.gpu_available:
            # With GPU, use more workers for faster data loading
            return min(4, self.device_info.get("logical_cores", 4))
        else:
            # With CPU, use fewer workers to avoid overhead
            return min(2, self.device_info.get("logical_cores", 2))
    
    def get_training_recommendations(self, recipe: dict) -> Dict:
        """
        Get training recommendations based on device and recipe.
        
        Args:
            recipe: Experiment recipe with hyperparameters
            
        Returns:
            Dictionary with recommendations
        """
        
        model_arch = recipe.get("model_architecture", "").lower()
        batch_size = recipe.get("batch_size", 32)
        
        recommendations = {
            "device": str(self.device),
            "device_type": self.device_info["type"],
            "original_batch_size": batch_size,
            "recommended_batch_size": self.optimize_batch_size(batch_size, model_arch),
            "num_workers": self.optimize_num_workers(),
            "mixed_precision": self.gpu_available and self.cuda_version >= "11.0",
            "pin_memory": self.gpu_available,
            "use_compile": False  # torch.compile requires PyTorch 2.0+
        }
        
        # Check PyTorch version for torch.compile
        try:
            torch_version = tuple(map(int, torch.__version__.split('+')[0].split('.')))
            if torch_version >= (2, 0, 0) and self.gpu_available:
                recommendations["use_compile"] = True
        except:
            pass
        
        return recommendations
    
    def clear_gpu_cache(self):
        """Clear GPU cache if available."""
        if self.gpu_available:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def get_docker_gpu_config(self) -> Dict:
        """
        Get Docker GPU configuration for container.
        
        Returns:
            Dict with Docker GPU settings
        """
        
        if not self.gpu_available:
            return {"runtime": None, "device_requests": []}
        
        try:
            # Check if nvidia-docker is available
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if "nvidia" in result.stdout.lower():
                return {
                    "runtime": "nvidia",
                    "device_requests": [
                        {
                            "Driver": "nvidia",
                            "Count": -1,  # All GPUs
                            "Capabilities": [["gpu"]]
                        }
                    ],
                    "environment": {
                        "NVIDIA_VISIBLE_DEVICES": "all"
                    }
                }
            else:
                logger.warning("nvidia-docker runtime not available")
                return {"runtime": None, "device_requests": []}
                
        except Exception as e:
            logger.warning(f"Could not detect nvidia-docker: {e}")
            return {"runtime": None, "device_requests": []}


# Global instance
device_manager = DeviceManager()


def get_device_config_for_training(recipe: dict) -> str:
    """
    Generate device configuration code for training script.
    
    Args:
        recipe: Experiment recipe
        
    Returns:
        Python code string for device setup
    """
    
    recommendations = device_manager.get_training_recommendations(recipe)
    
    code = f"""
# ============================================
# DEVICE CONFIGURATION
# ============================================

import torch
import os

# Detect device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"   GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}} GB")
    
    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    # Clear cache
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU (GPU not available)")
    
    # CPU optimizations
    torch.set_num_threads(os.cpu_count())
    print(f"   CPU threads: {{os.cpu_count()}}")

# Optimized settings based on device
BATCH_SIZE = {recommendations['recommended_batch_size']}
NUM_WORKERS = {recommendations['num_workers']}
PIN_MEMORY = {recommendations['pin_memory']}

print(f"   Batch size: {{BATCH_SIZE}}")
print(f"   Num workers: {{NUM_WORKERS}}")
print(f"   Pin memory: {{PIN_MEMORY}}")
"""
    
    return code


def get_dataloader_config(recommendations: Dict) -> str:
    """Generate DataLoader configuration code."""
    
    return f"""
# DataLoader with optimized settings
trainloader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0
)

testloader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0
)
"""


# Example usage in config.py
def update_config_for_device():
    """Update config based on detected device."""
    import config
    
    device_info = device_manager.get_device_info()
    
    if device_info["type"] == "CPU":
        # Reduce timeouts and batch sizes for CPU
        config.DOCKER_RUN_TIMEOUT = 3600  # 1 hour
        logger.info("CPU detected: Extended timeouts for training")
    else:
        # GPU available - can use shorter timeouts
        config.DOCKER_RUN_TIMEOUT = 1800  # 30 minutes
        logger.info("GPU detected: Using standard timeouts")