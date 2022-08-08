#! /bin/bash

gradle clean
./gradlew build -PskipCheckerFramework --rerun-tasks
#java -jar jpf/jpf-core/build/RunJPF.jar simple-datasets/build/classes/Main.class
#java -jar jpf/jpf-core/build/RunJPF.jar simple-datasets/src/main/java/Main.jpf
java -jar jpf/jpf-core/build/RunJPF.jar +classpath=simple-datasets/build/classes/java/main simple-datasets/src/main/java/Main.jpf