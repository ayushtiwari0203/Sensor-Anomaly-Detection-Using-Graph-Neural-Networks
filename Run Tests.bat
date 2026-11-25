@echo off
title Run MTM909 Tests
cd /d "%~dp0"
echo ========================================================
echo   Running Automated Test Suite for Sensor GNN
echo ========================================================
echo.
pytest -v
echo.
pause
