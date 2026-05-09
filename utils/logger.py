"""简易日志工具"""
import time

def log_info(msg):
    print(f"[{time.strftime('%H:%M:%S')}] INFO: {msg}")

def log_warn(msg):
    print(f"[{time.strftime('%H:%M:%S')}] WARN: {msg}")

def log_error(msg):
    print(f"[{time.strftime('%H:%M:%S')}] ERROR: {msg}")