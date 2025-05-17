@echo off
echo Setting up EPL Prediction System...

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
mkdir data\raw
mkdir data\processed
mkdir models
mkdir visualizations

echo Setup complete!
echo.
echo To run the system:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run the main script: python main.py --all
echo 3. Or run the Streamlit app: streamlit run app.py
echo.
pause