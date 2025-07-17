import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
from torchvision import transforms
import rawpy
import numpy as np
from PIL import Image
import glob
import time

# 全局分辨率变量
SIZE = (1024, 768)  # (宽度, 高度)


# 简化版生成器
class SimpleGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # 下采样
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 中间层
        self.mid = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        # 上采样
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU()
        )

        # 输出层
        self.out = nn.Sequential(
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        m = self.mid(d2)
        u1 = self.up1(m)
        u2 = self.up2(u1)
        return self.out(u2)


def process_single_image(input_path, output_path, gen, device, transform):
    """处理单个图像文件"""
    try:
        # 加载输入图像
        if input_path.lower().endswith(('.nef', '.raw')):
            print(f"处理RAW文件: {os.path.basename(input_path)}")
            with rawpy.imread(input_path) as raw:
                rgb = raw.postprocess()
            input_img = Image.fromarray(rgb)
            # 确保是RGB模式
            if input_img.mode != 'RGB':
                input_img = input_img.convert('RGB')
        else:
            print(f"处理图像文件: {os.path.basename(input_path)}")
            input_img = Image.open(input_path)
            # 确保是RGB模式
            if input_img.mode != 'RGB':
                input_img = input_img.convert('RGB')

        # 保存原始尺寸
        original_size = input_img.size
        print(f"原始图像尺寸: {original_size}")

        # 预处理
        img_tensor = transform(input_img).unsqueeze(0).to(device)

        # 预测
        with torch.no_grad():
            output = gen(img_tensor)

        # 后处理 - 修复彩色输出问题
        # 1. 将张量从GPU移到CPU
        output = output.squeeze(0).cpu()
        # 2. 反归一化
        output = output * 0.5 + 0.5
        # 3. 转换为numpy数组并调整维度顺序
        output = output.numpy().transpose(1, 2, 0)
        # 4. 转换为0-255范围
        output = (output * 255).clip(0, 255).astype(np.uint8)

        # 创建输出图像
        output_img = Image.fromarray(output, 'RGB')
        # 调整到原始尺寸
        output_img = output_img.resize(original_size)

        # 保存结果
        output_img.save(output_path)
        print(f"处理完成! 结果已保存到: {output_path}\n")

        return True

    except Exception as e:
        print(f"处理图像 {os.path.basename(input_path)} 时出错: {e}\n")
        return False


def batch_process_images(input_dir, output_dir, model_path):
    """批量处理目录中的所有图像"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    gen = SimpleGenerator()
    try:
        gen.load_state_dict(torch.load(model_path, map_location=device))
        gen.to(device)
        gen.eval()
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 支持的图像格式
    image_extensions = ['.nef', '.NEF', '.raw', '.RAW',
                        '.jpg', '.jpeg', '.JPG', '.JPEG',
                        '.png', '.PNG', '.tiff', '.TIFF']

    # 获取所有支持的图像文件 - 修复重复读取问题
    input_files = []
    for ext in image_extensions:
        # 使用精确匹配，避免重复
        pattern = os.path.join(input_dir, f"*{ext}")
        files = glob.glob(pattern)
        # 过滤掉已添加的文件
        new_files = [f for f in files if f not in input_files]
        input_files.extend(new_files)

    # 打印找到的文件列表
    print("\n找到的文件列表:")
    for file in input_files:
        print(f"  - {os.path.basename(file)}")

    if not input_files:
        print(f"\n在 {input_dir} 中没有找到支持的图像文件")
        return

    print(f"\n找到 {len(input_files)} 个待处理图像文件")
    print("开始批量处理...\n")

    # 统计处理结果
    success_count = 0
    fail_count = 0

    # 记录开始时间
    start_time = time.time()

    # 处理每个文件
    for i, input_path in enumerate(input_files):
        print(f"处理进度: {i + 1}/{len(input_files)}")

        # 创建输出路径
        filename = os.path.basename(input_path)
        # 替换扩展名为.jpg
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base_name}.jpg")

        # 处理单个图像
        if process_single_image(input_path, output_path, gen, device, transform):
            success_count += 1
        else:
            fail_count += 1

    # 计算处理时间
    total_time = time.time() - start_time
    avg_time = total_time / len(input_files) if input_files else 0

    print("\n" + "=" * 50)
    print(f" 批量处理完成!")
    print(f" 总处理文件数: {len(input_files)}")
    print(f" 成功处理数: {success_count}")
    print(f" 失败处理数: {fail_count}")
    print(f" 总耗时: {total_time:.2f}秒")
    print(f" 平均每张耗时: {avg_time:.2f}秒")
    print("=" * 50)


# 主预测函数
def main():
    # ========== 配置参数 ========== #
    # 输入图像目录
    input_dir = "./new_images"
    # 输出图像目录 (修改为./results)
    output_dir = "./results"
    # 生成器模型路径
    model_path = "models/generator_epoch_9.pth"
    # ============================= #

    print("\n" + "=" * 50)
    print(" 开始批量图像转换")
    print(f" 输入目录: {input_dir}")
    print(f" 输出目录: {output_dir}")
    print(f" 模型路径: {model_path}")
    print("=" * 50 + "\n")

    batch_process_images(input_dir, output_dir, model_path)


if __name__ == "__main__":
    main()
