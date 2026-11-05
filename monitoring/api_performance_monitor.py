
"""
API Performance Monitoring
Lab 12 - Experiment 12.2

Tracks request count, success/failure rate,
response latency, and throughput.
"""

from collections import deque
from threading import Lock
from time import perf_counter

lock = Lock()

total_requests = 0
successful_requests = 0
failed_requests = 0
total_latency_ms = 0.0
max_latency_ms = 0.0

request_timestamps = deque()
first_request_time = None


def record_request(status_code, latency_ms):
    global total_requests
    global successful_requests
    global failed_requests
    global total_latency_ms
    global max_latency_ms
    global first_request_time

    now = perf_counter()

    with lock:
        total_requests += 1
        total_latency_ms += latency_ms
        max_latency_ms = max(max_latency_ms, latency_ms)

        if 200 <= status_code < 400:
            successful_requests += 1
        else:
            failed_requests += 1

        if first_request_time is None:
            first_request_time = now

        request_timestamps.append(now)

        # Keep only the last 60 seconds of timestamps
        while request_timestamps and now - request_timestamps[0] > 60:
            request_timestamps.popleft()


def get_performance_summary():
    now = perf_counter()

    with lock:
        elapsed = (
            min(60, now - first_request_time)
            if first_request_time is not None
            else 0
        )

        recent_requests = len(request_timestamps)

        throughput = (
            recent_requests / elapsed
            if elapsed > 0
            else 0
        )

        average_latency = (
            total_latency_ms / total_requests
            if total_requests > 0
            else 0
        )

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "average_latency_ms": round(average_latency, 3),
            "maximum_latency_ms": round(max_latency_ms, 3),
            "requests_per_second": round(throughput, 3),
            "requests_in_last_60_seconds": recent_requests
        }
