#!/bin/bash

HOME="/Users/nadeeshan/Desktop/TOSEM/complexity-verification-project"
METHOD_DIR="$HOME/dataset4/code/methods"
OUTPUT_DIR="$HOME/dataset4/code/results"

mkdir -p "$OUTPUT_DIR"

i=1

# Use `find` to handle filenames with spaces reliably.
# -type f limits the search to regular files.
# -print0 terminates each filename with a null character.
# The `while` loop reads the null-terminated filenames safely.
for entry in "$METHOD_DIR"/*; do
    echo "Processing $entry..."
    filename=$(basename $entry)
    
    # Use double quotes for variable expansion to handle filenames with spaces.
    # Use `>>` to append, or use `>` if you want to overwrite each time.
    # The output filename will be correctly indexed.
    cbmc "$entry" --smt2 --memory-leak-check --memory-cleanup-check --unsigned-overflow-check  --pointer-overflow-check --conversion-check --float-overflow-check --nan-check --enum-range-check --retain-trivial-checks --unwind 5 --trace > "$OUTPUT_DIR/${filename}_results.txt"
    
    # Increment counter
    ((i++))
done

mkdir "$OUTPUT_DIR/merged"
cat "$OUTPUT_DIR"/* > $OUTPUT_DIR/merged/merged.txt

echo "Script finished."