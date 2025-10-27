## read the dataset10/code/questions_modified.c file and extract methods
## search the string for text "// ID <number>" number starts from 0 to 127. 
## extract the text between "// ID <number>" and "// ID <number+1>" and save it to a file named method<number>.c
## if the method is the last method i.e. "// ID 127" then extract the text till the end of the file
## append int main() { method<number>(); return 0; } to each method file
## save all the method files to dataset10/code/methods

import os
import re
from typing import List

def extract_methods_from_file(source_file: str, output_dir: str) -> List[str]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(source_file, 'r') as f:
        content = f.read()

    # Regex to find method blocks
    method_pattern = re.compile(r'// ID (\d+)(.*?)(?=// ID \d+|$)', re.DOTALL)
    matches = method_pattern.findall(content)

    method_files = []
    for method_id, method_body in matches:
        method_id = int(method_id)
        method_filename = os.path.join(output_dir, f'method{method_id}.c')
        with open(method_filename, 'w') as mf:
            mf.write('#include <stdio.h>\n\n')
            mf.write(method_body.strip() + '\n\n')
            mf.write(f'int main() {{ method{method_id}(); return 0; }}\n')
        method_files.append(method_filename)
        print(f'Extracted method ID {method_id} to {method_filename}')

    return method_files

if __name__ == "__main__":
    source_file = 'dataset4/code/modified.c'
    output_dir = 'dataset4/code/methods'
    extract_methods_from_file(source_file, output_dir)