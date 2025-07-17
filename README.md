# 图像风格转换项目

这是一个基于PyTorch实现的图像风格转换项目，能够将专业相机拍摄的RAW格式图像转换为具有特定风格的JPEG图像。项目包含完整的训练和预测流程，支持从检查点恢复训练，适合专业摄影师和图像处理爱好者使用。

## 项目结构

```bash
image-style-transfer/
├── train.py               # 训练脚本（支持恢复训练）
├── predict.py             # 批量预测脚本
├── models/                # 保存训练好的模型
│   └── generator_epoch_*.pth
├── checkpoints/           # 训练检查点（模型+优化器+轮数）
│   └── checkpoint_epoch_*.pth
├── images/                # 训练过程中的结果示例
│   └── result_epoch_*/
│   └── first/             # 训练用的原始图像
│   └── end/               # 训练用的处理过后的图像
├── new_images/            # 待转换的原始图像
├── result/                # 转换后的图像输出
└── README.md              # 项目文档
```

## 功能特性

- **RAW格式支持**：直接处理NEF/RAW等专业相机格式
- **灵活训练**：
  - 从零开始训练新模型
  - 从任意检查点恢复训练
  - 自动保存最佳模型和训练状态
- **批量处理**：一键转换整个目录的图像
- **高质量输出**：保持原始图像分辨率
- **可视化监控**：每轮训练保存结果示例

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- rawpy (用于处理RAW格式)
- Pillow
- numpy

## 安装依赖

```bash
pip install torch torchvision rawpy pillow numpy
```

## 快速开始

### 1. 训练模型

```bash
python train.py
```

**配置文件 (train.py 顶部)**:
```python
# 原始图像目录
first_dir = "./images/first"
# 目标图像目录
end_dir = "./images/end"
# 总训练轮数
total_epochs = 10
# 恢复训练设置
resume_training = True  # 是否从检查点恢复训练
resume_checkpoint = None  # 指定检查点路径或自动使用最新
```

### 2. 批量转换图像

```bash
python predict.py
```

**配置文件 (predict.py 顶部)**:
```python
# 输入图像目录
input_dir = "./new_images"
# 输出图像目录
output_dir = "./result"
# 生成器模型路径
model_path = "models/generator_epoch_9.pth"
```

## 详细使用指南

### 数据准备

1. 创建两个目录：
   - `./images/first`: 存放原始NEF/RAW文件
   - `./images/end`: 存放对应的目标JPEG图像

2. 文件命名规范：
   - 原始文件：`DSC_5171.nef`, `DSC_5172.nef`, ...
   - 目标文件：`100030.jpg`, `100031.jpg`, ...

### 训练模型

1. **首次训练**:
   - 设置 `resume_training = False`
   - 运行 `python train.py`

2. **恢复训练**:
   - 设置 `resume_training = True`
   - (可选) 指定 `resume_checkpoint = "checkpoints/checkpoint_epoch_5.pth"`
   - 运行 `python train.py`

### 批量转换图像

1. 将需要转换的图像放入 `new_images/` 目录
2. 修改 `predict.py` 中的模型路径
3. 运行 `python predict.py`
4. 转换后的图像将保存在 `result/` 目录

### 支持的图像格式

- 输入格式: NEF, RAW, JPG, JPEG, PNG, TIFF
- 输出格式: JPG

## 高级配置

### 训练参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `SIZE` | (1024, 768) | 训练分辨率 (宽, 高) |
| `batch_size` | 1 | 训练批次大小 |
| `lr` | 2e-4 | 学习率 |
| `save_interval` | 3 | 检查点保存间隔 (轮数) |

### 预测参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `input_dir` | "./new_images" | 输入图像目录 |
| `output_dir` | "./result" | 输出图像目录 |
| `model_path` | "models/generator_epoch_2.pth" | 生成器模型路径 |

## 常见问题

### 1. 训练过程中出现内存不足

解决方案：
- 减小 `batch_size`
- 降低训练分辨率 (`SIZE`)
- 使用更小的模型

### 2. 输出的图像颜色异常

解决方案：
- 确保输入图像是RGB模式
- 检查数据预处理流程
- 验证模型是否训练充分

### 3. 文件匹配问题

解决方案：
- 确认原始文件和目标文件命名符合规范
- 检查 `create_file_mapping` 函数中的匹配逻辑
- 确保文件扩展名大小写一致

## 结果示例

训练过程中每轮生成的结果示例保存在 `images/result_epoch_*/` 目录：
- `input.jpg`: 原始图像
- `generated.jpg`: 生成图像
- `target.jpg`: 目标图像

## 性能优化

- **GPU加速**：自动检测并使用CUDA
- **批量处理**：一次处理整个目录的图像
- **高效存储**：检查点只保存必要状态


## 联系方式

如有任何问题，请联系：jia_hua_wu@yeah.net

---

**提示**：首次训练建议使用小规模数据集测试，待流程验证无误后再进行完整训练。
