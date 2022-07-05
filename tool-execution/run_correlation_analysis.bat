cd ..
call complexity_verification_project_venv\Scripts\activate.bat
call complexity_verification_project_venv\Scripts\python.exe parser.py
call complexity_verification_project_venv\Scripts\python.exe correlation.py
pause