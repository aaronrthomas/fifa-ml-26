@echo off
echo ================================================================
echo  FIFA World Cup 2026 - Knockout Stage Prediction System
echo ================================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install numpy scipy pandas scikit-learn xgboost lightgbm catboost requests beautifulsoup4 lxml tqdm joblib statsmodels --quiet
if errorlevel 1 (
    echo WARNING: Some packages failed. Trying minimal install...
    pip install numpy scipy pandas scikit-learn --quiet
)

echo.
echo [2/3] Running standalone predictor (fast, no heavy ML packages needed)...
python generate_submission_direct.py

echo.
echo [3/3] Optionally run the full ML pipeline (requires all packages):
echo   python main.py --offline --no-ml-validation
echo   python main.py --offline  (includes 6-model ML comparison)
echo.
echo ================================================================
echo  OUTPUTS:
echo    predictions\submission.csv  (competition submission file)
echo    predictions\prediction_report.md  (full analysis)
echo ================================================================
pause
