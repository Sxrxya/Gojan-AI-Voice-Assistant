@echo off
echo.
echo =============================================
echo  GOJAN AI ASSISTANT — RUNNING ALL TESTS
echo =============================================
echo.
cd /d "c:\Users\welcome\Downloads\Gojan AI\Gojan-AI-Voice-Assistant"
call venv\Scripts\activate
python test_all_components.py
echo.
pause
