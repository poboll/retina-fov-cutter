"""
视网膜图像脱敏处理工具（支持YAML配置）
"""
import cv2
import yaml
import argparse
import numpy as np
import os
import pandas as pd
import shutil
import logging
from datetime import datetime
from typing import Optional, Tuple
# 配置类型提示
class ProcessingConfig:
    def __init__(self, config_dict: dict):
        self.retry_attempts = config_dict.get('retry_attempts', 3)
        self.extension_ratio = config_dict.get('extension_ratio', 1.2)
        self.min_radius = config_dict.get('min_radius', 256)
        self.max_radius = config_dict.get('max_radius', 0)
class IOConfig:
    def __init__(self, config_dict: dict):
        self.input_dir = config_dict.get('input_dir', './data/raw')
        self.output_dir = config_dict.get('output_dir', './data/processed')
        self.log_file = config_dict.get('log_file', 'processing.log')
class AppConfig:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            
        self.processing = ProcessingConfig(config_data.get('processing', {}))
        self.io = IOConfig(config_data.get('io', {}))
def configure_logging(log_path: str) -> None:
    """配置日志记录系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
def crop_and_extend(
    image_path: str,
    output_path: str,
    config: ProcessingConfig
) -> Optional[np.ndarray]:
    """
    核心处理逻辑（已参数化配置）
    
    :param image_path: 输入图像路径
    :param output_path: 输出路径
    :param config: 处理配置参数
    :return: 处理后的图像数组或None
    """
    attempt = 0
    while attempt < config.retry_attempts:
        # 图像读取容错
        if not os.path.exists(image_path):
            logging.error(f"文件不存在: {image_path}")
            return None
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"图像读取失败: {image_path} (尝试 {attempt + 1})")
            attempt += 1
            continue
        # 图像处理流程
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            circles = cv2.HoughCircles(
                edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                param1=50, param2=30, 
                minRadius=config.min_radius,
                maxRadius=config.max_radius
            )
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                largest = max(circles, key=lambda x: x[2])
                x, y, r = largest
                
                # 动态计算扩展高度
                new_height = int(2 * r * config.extension_ratio)
                start_h = max(0, y - new_height // 2)
                end_h = min(img.shape[0], y + new_height // 2)
                
                cropped = img[start_h:end_h, :]
                cv2.imwrite(output_path, cropped)
                return cropped
            else:
                logging.warning(f"未检测到有效圆形: {image_path}")
                attempt += 1
        except Exception as e:
            logging.error(f"处理异常: {str(e)}")
            attempt += 1
    # 失败处理策略
        exit(1)