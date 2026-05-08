"""
音频采集模块
基于 PyAudio 实现实时音频采集，采用双线程 + 线程安全队列架构。
音频采集线程只做最简数据搬运，处理线程累积数据到 1 秒后触发回调。
支持多设备管理、静音检测、滑动窗口等扩展。
"""

import pyaudio
import numpy as np
import threading
import queue
import time
import sys
from utils.logger import log_info, log_warn
import config


class AudioCapture:
    """
    实时音频采集器。
    实例化后调用 start(callback) 开始采集，每攒够 1 秒音频时调用 callback。
    调用 stop() 释放资源。
    """
    def __init__(self, device_index=None):
        self.device_index = device_index
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.queue = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
        self.buffer = b""                      # 处理线程的字节累积区
        self.stop_event = threading.Event()
        self.segment_callback = None           # 每满 1 秒触发的外部函数
        self.process_thread = None

        # 打印可用设备供选择
        self._print_devices()

    def _print_devices(self):
        """列出所有音频输入设备"""
        log_info("==== 可用音频输入设备 ====")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                log_info(f"[{i}] {info['name']} (采样率: {info['defaultSampleRate']:.0f}Hz)")
        log_info("===========================")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio 回调函数（运行在音频线程）。
        只将原始字节放入队列，立即返回，避免阻塞导致 xrun。
        """
        if status:
            log_warn(f"音频状态异常: {status}")
        # 队列满时丢弃最旧数据，保证系统不崩溃
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put(in_data)
        return (None, pyaudio.paContinue)

    def start(self, segment_callback):
        """
        启动音频采集。
        Args:
            segment_callback: 函数 func(audio_segment: np.ndarray, timestamp: float)
                              audio_segment 为 int16 型 numpy 数组，长度 16000。
        """
        self.segment_callback = segment_callback
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=config.CHANNELS,
            rate=config.SAMPLE_RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=config.FRAMES_PER_BUFFER,
            stream_callback=self.audio_callback
        )
        log_info("音频流已启动...")
        self.stop_event.clear()
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

    def _process_loop(self):
        """处理线程：从队列累积数据，满 1 秒调用回调。"""
        bytes_per_segment = config.SEGMENT_SIZE * config.CHANNELS * config.SAMPLE_WIDTH
        while not self.stop_event.is_set():
            try:
                data = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self.buffer += data
            # 只要缓冲区够 1 秒，就生成 numpy 数组并回调
            while len(self.buffer) >= bytes_per_segment:
                segment_bytes = self.buffer[:bytes_per_segment]
                self.buffer = self.buffer[bytes_per_segment:]
                segment = np.frombuffer(segment_bytes, dtype=np.int16)
                timestamp = time.time()
                if self.segment_callback:
                    self.segment_callback(segment, timestamp)

    def stop(self):
        """停止采集，释放资源。"""
        self.stop_event.set()
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        log_info("音频采集已停止。")