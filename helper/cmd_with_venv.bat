@echo off
set "venv_dir=venv"

REM Check if the virtual environment directory exists
if exist "%venv_dir%" (
    echo Virtual environment already exists.
) else (
    echo Creating virtual environment...
    python -m venv %venv_dir%
    echo Virtual environment created.

    REM Install the requirements
    call "%venv_dir%\Scripts\activate"
    if exist requirements.txt (
        echo Installing requirements...
        pip install -r requirements.txt
        echo Requirements installed.
    ) else (
        echo No requirements.txt file found.
    )
)

REM Activate the virtual environment and open a new Command Prompt
call "%venv_dir%\Scripts\activate"
start cmd /k call "%venv_dir%\Scripts\activate"
