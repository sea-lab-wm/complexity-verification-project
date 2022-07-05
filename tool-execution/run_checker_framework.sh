#! /bin/bash

./gradlew build --rerun-tasks 2> data/checker_framework_output.txt
#java -jar build\libs\complexity-verification-project.jar

echo "Checker Framework successfuly ran. View output in data/checker_framework_output.txt"