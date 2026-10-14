@echo off
chcp 65001 >nul
title School Logo Detection
cd /d D:\school_logo_detect
"C:\Users\Administrator\anaconda3\envs\school_logo\python.exe" camera_detect.py
echo.
echo ===== exit code: %ERRORLEVEL% =====
pause
