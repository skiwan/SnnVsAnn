import os
import shutil

def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_temp_folder(base_folder):
    temp_path = os.path.join(base_folder, 'temp')
    create_folder_if_not_exists(temp_path)

def delete_temp_folder(base_folder):
    temp_path = os.path.join(base_folder, 'temp')
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

def delete_folder(folder_dir):
    if os.path.exists(folder_dir):
        shutil.rmtree(folder_dir)