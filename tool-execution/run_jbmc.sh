#! /bin/bash

if ! command -v jbmc &> /dev/null
then
    echo "jmbc could not be found"
    exit
fi

( cd simple-datasets/src/main/java && javac Main.java )
( cd simple-datasets/src/main/java && jbmc Main --unwind 5 > ../../../../data/jbmc_output.txt )