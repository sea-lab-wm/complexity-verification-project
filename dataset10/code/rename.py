## read the questions.c file and rename the main() to method<i> i = 1,2,3...
import os
import re
import shutil
import sys

def read_the_file(path):
    with open(path, 'r') as file:
        data = file.read()
    return data

def search_main_term(data):
    original_data = data
    starting_index = 0
    modified_data = data
    main_count = 0

    while True:
        main_index = modified_data.find('main()', starting_index)
        if main_index == -1:
            break
        main_count += 1
        new_main = f'method{main_count}()'
        modified_data = modified_data[:main_index] + new_main + modified_data[main_index + len('main()'):]
        starting_index = main_index + len(new_main)
    
    return modified_data



def main():
   path = "/Users/nadeeshan/Desktop/TOSEM/complexity-verification-project/dataset10/code/questions.c"
   output_path = "/Users/nadeeshan/Desktop/TOSEM/complexity-verification-project/dataset10/code/questions_modified.c"
   data = read_the_file(path)
   modified_data = search_main_term(data)
   with open(output_path, 'w') as file:
        file.write(modified_data)


main()
