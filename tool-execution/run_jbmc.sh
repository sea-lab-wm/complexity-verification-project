#! /bin/bash

( cd simple-datasets/src/main/java && javac Main.java )
( cd simple-datasets/src/main/java && jbmc Main --unwind 20 > ../../../../data/jbmc_output.txt )