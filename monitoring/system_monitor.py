
"""
System Resource Monitoring
Lab 12 - Experiment 12.3
"""

import os
import psutil
from datetime import datetime, timezone


def get_system_metrics():

    process = psutil.Process(os.getpid())
    memory = psutil.virtual_memory()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "system_memory_percent": memory.percent,
        "available_memory_mb": round(
            memory.available / (1024 * 1024), 2
        ),
        "process_memory_mb": round(
            process.memory_info().rss / (1024 * 1024), 2
        ),
        "status": "active"
    }
