cd ..
call gradlew build --rerun-tasks 2> data\checker_framework_output.txt
::call java -jar build\libs\complexity-verification-project.jar
pause