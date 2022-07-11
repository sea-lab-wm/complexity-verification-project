#! /bin/bash

gradle clean
infer run --report-console-limit-reset -- ./gradlew build -PskipCheckerFramework --rerun-tasks > data/infer_output.txt
#infer run -- ./gradlew build --rerun-tasks