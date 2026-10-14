@echo off
chcp 65001 >nul
title YOLOv8 train - school_logo
cd /d D:\school_logo_detect
"C:\Users\Administrator\anaconda3\envs\school_logo\python.exe" train.py
echo.
echo ===== exit code: %ERRORLEVEL% =====
pause
