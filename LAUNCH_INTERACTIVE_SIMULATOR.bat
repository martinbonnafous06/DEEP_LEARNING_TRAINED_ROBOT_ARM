@echo off
title LIBERO / MuJoCo 3D Franka Panda - Interactive World Model & Policy Tester
cd /d "%~dp0"
echo Launching Interactive LIBERO / MuJoCo 3D Franka Panda Simulator...
"C:\Users\marti\miniconda3\envs\libero\python.exe" scripts\09_interactive_mujoco_gui.py
pause
