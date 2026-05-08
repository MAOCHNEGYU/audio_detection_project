"""
项目环境初始化模块。
确保当前工作目录为项目根目录，并将根目录加入 sys.path，
防止因运行位置不同导致的相对路径错误。
"""

import os
import sys


def set_project_root():
    """
    将当前工作目录切换到项目根目录，并确保根目录在 Python 搜索路径中。
    该函数假设本文件位于 <root>/utils/project_init.py，
    项目根目录即 utils 的上一级目录。
    """
    # 获取本文件的绝对路径，再向上两级得到项目根目录
    current_file = os.path.abspath(__file__)          # .../utils/project_init.py
    utils_dir = os.path.dirname(current_file)         # .../utils
    project_root = os.path.dirname(utils_dir)         # .../

    # 切换工作目录
    os.chdir(project_root)

    # 将项目根目录添加到 sys.path（如果尚未加入）
    if project_root not in sys.path:
        sys.path.insert(0, project_root)