import os, shutil

def moving_folder(path, des):
    if not os.path.exists(path) or not os.path.exists(des):
        print(f"Error: Source folder '{path}' not found.")
        return
    try:
        shutil.move(path, des)
        print(f"Successfully moved folder '{os.path.basename(path)}' to '{des}'")
    except shutil.Error as e:
        print(f"There are errors. Try again!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
import zipfile
def unzip_file(path, des, delete=True):
    if not os.path.exists(path) or not os.path.exists(des):
        print(f"Error: Source folder '{path}' not found.")
        return
    with zipfile.ZipFile(path) as zipref:
        zipref.extractall(des)
        print(f"Extracted zipfile in {des}")
    if delete:
        os.remove(path)
        return