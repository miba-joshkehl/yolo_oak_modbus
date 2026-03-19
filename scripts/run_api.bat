@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0run_api.ps1" %*
pause