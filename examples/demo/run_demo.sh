#!/usr/bin/env bash
# End-to-end demo of the recommendation API using public research data (MovieLens ids).
#
#   ./examples/demo/run_demo.sh                    # API at http://localhost:8000
#   API_URL=http://host:port ./examples/demo/run_demo.sh
#
# Requires: curl, python3 and a running API with an approved model
# (python src/main.py --approve ... or the Docker trainer, see INSTALLATION_AND_SETUP.md).
set -euo pipefail
API_URL="${API_URL:-http://localhost:8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
J='content-type: application/json'
pp() { python3 -m json.tool --no-ensure-ascii; }
step() { printf '\n=== %s\n$ %s\n' "$1" "$2"; }

step "1. Health" "GET /health"
curl -sS "$API_URL/health" | pp

step "2. Model information" "GET /model/info"
curl -sS "$API_URL/model/info" | python3 "$HERE/summarise.py" info

step "3. Top-5 recommendations for a known user (SVD)" "GET /recommendations/1?n=5"
curl -sS "$API_URL/recommendations/1?n=5" | python3 "$HERE/summarise.py" recs

step "4. Unknown user: popularity fallback (cold start)" "GET /recommendations/999999?n=3"
curl -sS "$API_URL/recommendations/999999?n=3" | python3 "$HERE/summarise.py" recs

step "5. Same user, excluding the previous top movie (3347) and an unknown id" "POST /recommendations"
curl -sS -X POST "$API_URL/recommendations" -H "$J" \
  -d '{"user_id": 1, "n": 3, "exclude_rated": true, "exclude_movie_ids": [3347, 99999999]}' | python3 "$HERE/summarise.py" recs

step "6. Predict a rating for a user-movie pair" "POST /predict"
curl -sS -X POST "$API_URL/predict" -H "$J" -d '{"user_id": 1, "movie_id": 50}' | pp

step "7. Invalid input is rejected with HTTP 422" "GET /recommendations/0"
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' "$API_URL/recommendations/0"

step "8. Record an observed rating (delayed ground truth, used for retraining)" "POST /feedback"
curl -sS -X POST "$API_URL/feedback" -H "$J" -d '{"user_id": 1, "movie_id": 50, "rating": 4.5}' | pp

step "9. Monitoring summary" "GET /monitoring/summary"
curl -sS "$API_URL/monitoring/summary" | python3 "$HERE/summarise.py" summary

step "10. Prometheus metrics (first lines)" "GET /metrics"
curl -sS "$API_URL/metrics" | grep -E '^recsys_(requests_total|cold_start_total|feedback_total)' | head -8
printf '\nDemo finished.\n'
