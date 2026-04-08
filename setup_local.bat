@echo off
echo ═══════════════════════════════════════════════════════
echo   GOJAN AI VOICE ASSISTANT — LOCAL SETUP (CPU ONLY)
echo ═══════════════════════════════════════════════════════
echo.

REM --- Create virtual environment ---
echo [1/4] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    echo Make sure Python 3.10+ is installed and on PATH.
    pause
    exit /b 1
)
echo       Done.

REM --- Activate virtual environment ---
echo [2/4] Activating virtual environment...
call venv\Scripts\activate
python -m pip install --upgrade pip

REM --- Install llama-cpp-python with CPU flags ---
echo [3/4] Installing llama-cpp-python (CPU only)...
set CMAKE_ARGS=-DLLAMA_CUBLAS=off
pip install llama-cpp-python==0.2.56
if errorlevel 1 (
    echo WARNING: llama-cpp-python install failed.
    echo You may need Visual Studio Build Tools installed.
    echo Try: pip install llama-cpp-python==0.2.56 --prefer-binary
)

REM --- Install remaining dependencies ---
echo [4/4] Installing remaining dependencies...
pip install -r requirements_local.txt
echo.

REM --- Check required files ---
echo ═══════════════════════════════════════════════════════
echo   CHECKING REQUIRED FILES
echo ═══════════════════════════════════════════════════════
echo.

if exist "models\gguf\gojan_ai_q4.gguf" (
    echo   [OK] models\gguf\gojan_ai_q4.gguf
) else (
    echo   [WARNING] models\gguf\gojan_ai_q4.gguf NOT FOUND
    echo            Download from Colab and place here.
)

if exist "vector_db\college.index" (
    echo   [OK] vector_db\college.index
) else (
    echo   [WARNING] vector_db\college.index NOT FOUND
    echo            Download from Colab and place here.
)

if exist "vector_db\documents.pkl" (
    echo   [OK] vector_db\documents.pkl
) else (
    echo   [WARNING] vector_db\documents.pkl NOT FOUND
    echo            Download from Colab and place here.
)

echo.
echo ═══════════════════════════════════════════════════════
echo   SETUP COMPLETE!
echo ═══════════════════════════════════════════════════════
echo.
echo   To run the assistant:
echo     1. Activate the venv:  venv\Scripts\activate (CMD) or .\venv\Scripts\Activate.ps1 (PowerShell)
echo     2. Run:                cd phase_b_local
echo                            python main.py
echo.
pause
