"""项目环境初始化：确保工作目录为项目根目录"""
import os
import sys

def set_project_root():
    current_file = os.path.abspath(__file__)          # .../utils/project_init.py
    utils_dir = os.path.dirname(current_file)         # .../utils
    project_root = os.path.dirname(utils_dir)         # .../
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)