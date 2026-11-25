@echo off
title Sensor Anomaly Detection GNN
cd /d "%~dp0"
echo ========================================================
echo   Running Sensor Anomaly Detection using GNN (INT423 CA2)
echo ========================================================
echo.
python run_pipeline.py
echo.
pause
