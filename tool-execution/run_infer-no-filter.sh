#!/bin/bash

./gradlew clean
infer run --no-filtering --report-console-limit-reset -- ./gradlew build -PskipCheckerFramework --rerun-tasks > data/infer-no-filter_output.txt
