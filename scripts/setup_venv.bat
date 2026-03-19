@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0setup_venv.ps1" %*
pause