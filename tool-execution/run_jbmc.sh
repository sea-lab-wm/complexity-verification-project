#! /bin/bash

( cd simple-datasets/src/main/java && javac Main.java )
( cd simple-datasets/src/main/java && jbmc Main > ../../../../data/jbmc_output.txt )