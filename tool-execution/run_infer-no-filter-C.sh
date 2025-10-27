#!/bin/bash

HOME="/home/kgdesilva/Desktop/TOSEM/complexity-verification-project"
OUTPUT_DIR="$HOME/dataset10/code/results-Infer"

mkdir -p "$OUTPUT_DIR"

infer run --no-filtering --report-console-limit-reset  -- clang "$HOME/dataset10/code/modified-Infer.c" -lm -o main > "$OUTPUT_DIR/Infer-results.txt"


echo "Script finished."