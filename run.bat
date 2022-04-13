call gradlew build --rerun-tasks 2> output.txt
::call java -jar build\libs\complexity-verification-project.jar
call complexity_verification_project_venv\Scripts\activate.bat
call complexity_verification_project_venv\Scripts\python.exe parser.py
pause