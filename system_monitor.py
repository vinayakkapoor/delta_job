# system_monitor.py
import logging
import platform
import threading
import time
from typing import Dict, Optional, Tuple
import psutil  # Install with: pip install psutil
import torch

class SystemMonitor:
    """Modular system resource monitor for GPU, CPU, and RAM"""
    
    def __init__(
        self, 
        log_interval: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            log_interval: Seconds between measurements
            logger: Optional pre-configured logger
        """
        self._stop_event = threading.Event()
        self.log_interval = log_interval
        self.peak_stats: Dict[str, float] = {}
        
        # Configure logger
        self.logger = logger or self._configure_default_logger()
        
        # Initialize platform-specific properties
        self.os_name = platform.system()
        self.cpu_count = psutil.cpu_count(logical=False)
        self.logical_cpu_count = psutil.cpu_count(logical=True)
        
        # GPU capabilities
        self.cuda_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.cuda_available else 0

    def _configure_default_logger(self) -> logging.Logger:
        """Create a default logger if none provided"""
        logger = logging.getLogger("SystemMonitor")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU memory stats in GB"""
        stats = {}
        if self.cuda_available:
            for i in range(self.gpu_count):
                stats.update({
                    f"gpu_{i}_allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
                    f"gpu_{i}_reserved_gb": torch.cuda.memory_reserved(i) / 1e9,
                    f"gpu_{i}_utilization": torch.cuda.utilization(i),
                })
            stats["total_gpu_allocated_gb"] = sum(
                stats[f"gpu_{i}_allocated_gb"] 
                for i in range(self.gpu_count)
            )
        return stats

    def get_cpu_stats(self) -> Dict[str, float]:
        """Get CPU utilization percentages"""
        return {
            "cpu_utilization": psutil.cpu_percent(),
            "cpu_per_core": dict(enumerate(psutil.cpu_percent(percpu=True))),
        }

    def get_ram_stats(self) -> Dict[str, float]:
        """Get RAM usage in GB"""
        ram = psutil.virtual_memory()
        return {
            "ram_total_gb": ram.total / 1e9,
            "ram_available_gb": ram.available / 1e9,
            "ram_used_gb": ram.used / 1e9,
            "ram_used_percent": ram.percent,
        }

    def get_system_stats(self) -> Dict:
        """Get all available system statistics"""
        stats = {}
        
        try:
            stats.update(self.get_cpu_stats())
            stats.update(self.get_ram_stats())
            if self.cuda_available:
                stats.update(self.get_gpu_stats())
                
            # Update peak values
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    self.peak_stats[key] = max(
                        self.peak_stats.get(key, float('-inf')), 
                        value
                    )
                    
        except Exception as e:
            self.logger.warning(f"Failed to collect system stats: {str(e)}")
            
        return stats

    def _monitoring_loop(self):
        """Continuous monitoring thread"""
        self.logger.info("Starting system monitor")
        while not self._stop_event.is_set():
            stats = self.get_system_stats()
            self._log_stats(stats)
            self._stop_event.wait(self.log_interval)

    def _log_stats(self, stats: Dict):
        """Format and log system statistics"""
        log_entries = [
            f"System Stats || CPU: {stats.get('cpu_utilization', 0):.1f}%",
            f"RAM: {stats.get('ram_used_percent', 0):.1f}%",
        ]
        
        if self.cuda_available:
            log_entries.append(
                f"GPU Mem: {stats.get('total_gpu_allocated_gb', 0):.2f}GB"
            )
            
        self.logger.info(" | ".join(log_entries))
        self.logger.debug(f"Full stats: {stats}")

    def start(self):
        """Start monitoring thread"""
        if not self._stop_event.is_set():
            self.peak_stats = {}  # Reset peak values
            threading.Thread(target=self._monitoring_loop, daemon=True).start()

    def stop(self):
        """Stop monitoring and log peak values"""
        self._stop_event.set()
        self.logger.info("PEAK USAGE || " + " | ".join(
            [f"{k}: {v:.2f}" for k, v in self.peak_stats.items()]
        ))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Example usage:
if __name__ == "__main__":
    monitor = SystemMonitor(log_interval=2)
    monitor.start()
    
    try:
        # Simulate work
        for _ in range(5):
            time.sleep(1)
    finally:
        monitor.stop()