#!/bin/sh

excel_files=$(find . -name "*.xlsx")

for file in ${excel_files}; do
    # excel files are secretly zip files. This particular file contains personal
    # information. Via https://apple.stackexchange.com/questions/389904/how-can-i-clear-metadata-personal-data-from-an-excel-file-in-microsoft-365-for
    zip -d ${file} docProps/core.xml
done
