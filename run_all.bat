@echo off
setlocal
cd /d "%~dp0"

echo [1/4] Install requirements
python -m pip install -r requirements.txt
if errorlevel 1 goto :err

if not exist "data\hotels.csv" (
    echo [2/4] Download dataset
    python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv', 'data/hotels.csv')"
    if errorlevel 1 goto :err
) else (
    echo [2/4] Dataset already downloaded
)

if not exist "models\final_model.pkl" (
    echo [3/4] Run notebooks - takes a few minutes
    python scripts\run_notebooks.py
    if errorlevel 1 goto :err
) else (
    echo [3/4] Final model already built
)

echo [4/4] Start Streamlit
python -m streamlit run app\app.py
goto :eof

:err
echo FAILED
exit /b 1
