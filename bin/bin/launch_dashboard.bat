@echo off
REM Navigate to the root directory of the Git repository
cd %~dp0

REM Create a virtual environment in the root folder
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install required packages from requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

REM Run the Jupyter Notebook located at src/dashboard.ipynb
jupyter notebook "src/dashboard.ipynb"

REM Optional: Pause to keep the command window open
pause
