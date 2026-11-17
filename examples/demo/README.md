# API demo

`run_demo.sh` calls every public endpoint of a running API: health, model information,
recommendations for a known user (SVD) and an unknown user (popularity fallback), extra
exclusions, rating prediction, input validation (HTTP 422), feedback, monitoring summary and
Prometheus metrics. It uses MovieLens ids (public research data) and needs `curl` and `python3`.

```bash
python src/main.py --approve --approval-reason "demo"      # once, if no model is approved yet
uvicorn api:app --app-dir src --port 8000                  # or: docker compose up -d api
./examples/demo/run_demo.sh                                # API_URL=http://host:port to target another address
```

A captured run is in `demo_output_example.txt`.
