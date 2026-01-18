#!/bin/bash

# HoloRay Medical Video Labeler - Start Script
# Runs both backend and frontend simultaneously

echo "🚀 Starting HoloRay Medical Video Labeler..."
echo ""

# Kill any existing processes on ports 8000 and 5173
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

# Start backend
echo "📡 Starting backend (port 8000)..."
cd be && source venv/bin/activate && python main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend
echo "🎨 Starting frontend (port 5173)..."
cd fe && npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Services started!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both services"

# Handle shutdown
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Wait for processes
wait
