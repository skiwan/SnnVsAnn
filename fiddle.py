import os

file_directory = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(file_directory)
print(file_directory, parent_folder)