#! /bin/bash

if ! command -v jbmc &> /dev/null
then
    echo "jmbc could not be found"
    exit
fi

gradle clean
./gradlew build -PskipCheckerFramework --rerun-tasks

( cd simple-datasets/build/classes/java/main && jbmc Main --unwind 5 > ../../../../../data/jbmc_output_simple_datasets.txt )

( cd dataset6/build/classes/java/main && jbmc Main --unwind 5 > ../../../../../data/jbmc_output_dataset6.txt )

( cd dataset9/build/classes/java/main && jbmc Main --unwind 5 > ../../../../../data/jbmc_output_dataset9.txt )