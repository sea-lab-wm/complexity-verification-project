cd ..
call gradle clean
call infer run --report-console-limit-reset -- ./gradlew build -PskipCheckerFramework --rerun-tasks > data/infer_output.txt
pause