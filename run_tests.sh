#!/usr/bin/env bash
# Runs each task's test suite in its own folder (each task is a self-contained service).
set -euo pipefail
cd "$(dirname "$0")"
status=0
for task in task1_mcp_server task2_mcp_gateway task3_pii_stream task4_router; do
  echo "=== $task ==="
  (cd "$task" && python -m pytest -q -c ../pytest.ini --rootdir .) || status=1
done
exit $status
