#! /bin/bash

gradle clean
./gradlew build -PskipCheckerFramework --rerun-tasks

java -jar jpf/jpf-core/build/RunJPF.jar +classpath=simple-datasets/build/classes/java/main simple-datasets/src/main/java/Main.jpf
#java -jar jpf/jpf-core/build/RunJPF.jar +classpath=dataset6/build/classes/java/main dataset6/src/main/java/Main.jpf
#java -jar jpf/jpf-core/build/RunJPF.jar +classpath=dataset9/build/classes/java/main:dataset9/libs/web4j.jar dataset9/src/main/java/Main.jpf