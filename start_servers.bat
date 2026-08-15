@echo off
echo ==================================================
echo Starting Insulin Drug Synthesis Platform
echo ==================================================

echo 1. Starting Python Combined Backend Server (Port 5001)...
start "Backend Server" cmd /k "cd backend && python combined_server.py"

timeout /t 3 /nobreak >nul

echo 2. Starting Vite Frontend Dev Server (Port 5173)...
start "Vite Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting up:
echo   - Backend API:    http://localhost:5001
echo   - Frontend App:   http://localhost:5173
echo.
