# Detector Demo App

Run the demo stack:

```bash
docker compose -f docker-compose.demo.yml up --build
```

The app exposes `/work`, `/metrics`, and `POST /admin/fault` with JSON
`{"name":"latency","enabled":true}`. Supported faults are `latency`, `errors`,
and `dependency_down`.
