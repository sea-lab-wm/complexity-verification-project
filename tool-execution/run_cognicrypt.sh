#! /bin/bash

gradle clean
./gradlew build -PskipCheckerFramework --rerun-tasks

java -cp CogniCrypt/CryptoAnalysis-2.8.0-SNAPSHOT-jar-with-dependencies.jar crypto.HeadlessCryptoScanner --rulesDir $(pwd)/CogniCrypt/rules --appPath $(pwd)/simple-datasets/build/classes/java/main 
java -cp CogniCrypt/CryptoAnalysis-2.8.0-SNAPSHOT-jar-with-dependencies.jar crypto.HeadlessCryptoScanner --rulesDir $(pwd)/CogniCrypt/rules --appPath $(pwd)/dataset6/build/classes/java/main 
java -cp CogniCrypt/CryptoAnalysis-2.8.0-SNAPSHOT-jar-with-dependencies.jar crypto.HeadlessCryptoScanner --rulesDir $(pwd)/CogniCrypt/rules --appPath $(pwd)/dataset9/build/classes/java/main 
#--reportPath $(pwd)/CogniCrypt