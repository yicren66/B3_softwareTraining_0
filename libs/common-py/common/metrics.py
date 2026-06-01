"""
Prometheus metrics helpers for the Jujube platform.

Exports standard counters, histograms, and gauges that every service can
register with its own Prometheus HTTP endpoint.
"""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Recognition service metrics
# ---------------------------------------------------------------------------

recognition_request_counter = Counter(
    "jujube_recognition_requests_total",
    "Total number of image-recognition requests received.",
    labelnames=["service", "model_version"],
)

recognition_latency_histogram = Histogram(
    "jujube_recognition_latency_seconds",
    "End-to-end latency of image-recognition inferences.",
    labelnames=["service", "model_version"],
    buckets=(
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
    ),
)

# ---------------------------------------------------------------------------
# Knowledge-graph query metrics
# ---------------------------------------------------------------------------

kg_query_counter = Counter(
    "jujube_kg_queries_total",
    "Total number of knowledge-graph queries.",
    labelnames=["service", "query_type"],
)

kg_query_latency_histogram = Histogram(
    "jujube_kg_query_latency_seconds",
    "Latency of knowledge-graph queries.",
    labelnames=["service", "query_type"],
    buckets=(
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ),
)

# ---------------------------------------------------------------------------
# System-wide gauges
# ---------------------------------------------------------------------------

active_users_gauge = Gauge(
    "jujube_active_users",
    "Estimated number of concurrently active users.",
    labelnames=["service"],
)
