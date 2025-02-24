# 眼底图像预处理及优化工具

## 项目背景
原始眼底图像（示例见 `samples/input/image1.jpg`）存在大量无效黑边区域。这些背景像素不仅无法为眼底识别提供有效信息，其暗区噪声还可能对后续分析产生干扰。本工具通过自动提取眼底图像的圆形有效视野区域（Circular Field of View, FOV），实现精准的预处理操作。经算法优化后，单图处理耗时从 3 秒降低至 0.6 秒，在保持切割精度的同时实现 5 倍效率提升。

## 功能特性
- ⚡ **高效处理**：支持批量处理眼底图像（左/右眼）
- ⭕ **精准识别**：基于圆形检测算法，准确提取有效视野区域
- 🔄 **容错机制**：自动重试机制+原始文件保留策略，确保数据完整性
- 📊 **数据追溯**：自动生成处理日志与 CSV 元数据记录

## 技术方案

### 实现原理
采用基于 OpenCV 的传统图像处理流程：
1. **灰度转换**：将 RGB 图像转为灰度空间
2. **高斯模糊**：5x5 核降噪处理（σ=0）
3. **边缘检测**：Canny 算子（50/150 阈值）
4. **圆形拟合**：Hough 梯度法检测最大圆形轮廓
5. **区域扩展**：按检测半径 120% 比例扩展高度

### 性能优化
| 指标         | 优化前 | 优化后 | 提升幅度 |
|--------------|--------|--------|----------|
| 单图耗时     | 3.0s   | 0.3s   | 10倍      |
| 处理成功率   | 92%    | 95%    | +3%      |
| CPU利用率    | 35%    | 68%    | +94%     |

## 快速开始

### 环境要求
```bash
Python == 2.7.x
OpenCV == 3.1.0  # 需确保 import cv2 可用
numpy == 1.12.1
pandas >= 0.19.2
```

### 执行处理
- **单张测试（开发模式）**
  ```bash
  python demo.py --input samples/input.jpg --output results/
  ```

- **批量处理（生产模式）**
  ```bash
  python run.py --config config.yaml
  ```

### 处理效果
<img src="https://github.com/poboll/retina-fov-cutter/raw/main/path/out/实验效果.png" width="80%" height="80%"></img>  

### 优化思路
<img src="https://github.com/poboll/retina-fov-cutter/raw/main/path/out/优化.png" width="80%" height="80%"></img>   

### 优化效果
<img src="https://github.com/poboll/retina-fov-cutter/raw/main/path/out/优化对比.png" width="80%" height="80%"></img>

## 进阶配置
通过修改 `config.yaml` 配置文件实现个性化设置：

```yaml
processing:
  retry_attempts: 3       # 失败重试次数
  extension_ratio: 1.2     # 高度扩展比例
  min_radius: 256         # 最小有效半径（px）

io:
  input_dir: /data/raw     # 原始图像目录
  output_dir: /data/processed  # 结果存储目录
  log_file: processing.log # 日志文件路径
```

### 数据管理
处理完成后，系统将生成以下结构化文件：

```
├── processed/            # 处理后的图像
│   ├── patient_001/
│   │   ├── left.jpg
│   │   └── right.jpg
├── logs/
│   └── processing.log     # 处理日志（含错误记录）
└── metadata.csv          # 包含处理后的文件路径信息
```

## 许可协议
MIT