@echo off
title NeuroArchitect Launcher
echo [1/2] Menyiapkan Lingkungan Proyek...
call venv\Scripts\activate
echo [2/2] Menjalankan NeuroArchitect AI...
streamlit run app.py
pause