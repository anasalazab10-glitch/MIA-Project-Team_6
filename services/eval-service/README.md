# eval-service

Evaluation + observability (Langfuse) service.

## Endpoints
- GET /health
- GET /benchmark_info  (reads BENCHMARK_PATH, default: services/eval-service/data/benchmark_100.json)

## Env
- BENCHMARK_PATH (optional)
- LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST (optional; enable tracing later)
