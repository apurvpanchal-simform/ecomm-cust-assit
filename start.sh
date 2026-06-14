#!/bin/bash
set -e

# Export API Base URL for Chainlit
export API_BASE_URL="http://localhost:8000"

echo "Starting FastAPI backend..."
# Start FastAPI backend in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment to ensure backend starts before UI tries to connect
sleep 3

echo "Starting Chainlit frontend..."
# Start Chainlit frontend in the foreground
chainlit run ui/app.py --host 0.0.0.0 --port 8501
kill $BACKEND_PID
