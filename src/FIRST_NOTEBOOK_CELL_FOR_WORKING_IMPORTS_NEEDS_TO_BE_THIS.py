# PASTE THIS TO THE FIRST CELL OF THE NOTEBOOK IN ORDER TO HAVE WORKING IMPORTS
import sys
import os
current_dir = os.getcwd()
parent_parent_dir = os.path.abspath(os.path.join(current_dir, '../..')) # tweak so that you get dir of code project

sys.path.append(parent_parent_dir)










# -----------------------------------------------------------
# or this

import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
target_dir_name = 'ba_code_project'
while True:
    # Check if the target directory exists in the current directory
    potential_target = os.path.join(current_dir, target_dir_name)
    if os.path.isdir(potential_target):
        code_root_dir = potential_target
        break
    # Move one level up
    parent_dir = os.path.dirname(current_dir)
    # If we're at the root of the file system and still haven't found it, stop
    if parent_dir == current_dir:
        code_root_dir = None
        break
    current_dir = parent_dir
if code_root_dir:
    # Add the found target directory to sys.path
    sys.path.append(code_root_dir)
else:
    print(f'Target directory not found.')