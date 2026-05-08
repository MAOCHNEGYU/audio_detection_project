"""
简易日志工具
提供带时间戳的控制台打印，方便调试和记录系统状态。
"""

import time

def log_info(msg):
    """打印普通信息"""
    print(f"[{time.strftime('%H:%M:%S')}] INFO: {msg}")

def log_warn(msg):
    """打印警告信息"""
    print(f"[{time.strftime('%H:%M:%S')}] WARN: {msg}")

def log_error(msg):
    """打印错误信息"""
    print(f"[{time.strftime('%H:%M:%S')}] ERROR: {msg}")