import os
import re

# print(os.getcwd())
# path = input("Enter the directory path where you need to  rename: ")
for filename in os.listdir('.'):
    if filename.lower().endswith('.jpg'):
        if str(os.getcwd()).find('right') > -1:
            prefix = 'right_'
        else:
            prefix = 'wrong_'

        new_file_name = prefix + filename

        os.rename(filename, new_file_name)
