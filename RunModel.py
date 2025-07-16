import os
import torch
from torchvision import transforms
from PIL import Image
import rawpy
import numpy as np
import glob


# 加载训练好的模型
def load_generator(model_path, device):
    from model import UNetGenerator  # 导入之前定义的生成器

    # 创建模型实例 (需要与训练时相同的参数)
    gen = UNetGenerator().to(device)

    # 加载权重
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()
    print(f"加载生成器模型: {model_path}")
    return gen


# 处理单个NEF文件
def process_nef_file(nef_path, generator, device, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载和预处理NEF文件
    try:
        with rawpy.imread(nef_path) as raw:
            rgb = raw.postprocess()
        input_img = Image.fromarray(rgb)
        print(f"处理: {os.path.basename(nef_path)}, 原始尺寸: {input_img.size}")
    except Exception as e:
        print(f"处理NEF失败 {nef_path}: {e}")
        return

    # 应用转换
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),  # 使用训练时的相同分辨率
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    input_tensor = transform(input_img).unsqueeze(0).to(device)
    print(f"输入张量尺寸: {input_tensor.shape}")

    # 生成图像
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # 后处理: 转换为PIL图像
    output_img = output_tensor.squeeze(0).cpu().detach()
    output_img = output_img.permute(1, 2, 0).numpy()
    output_img = (output_img * 0.5 + 0.5) * 255
    output_img = output_img.clip(0, 255).astype(np.uint8)
    output_img = Image.fromarray(output_img)

    # 保存结果
    base_name = os.path.splitext(os.path.basename(nef_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_generated.jpg")
    output_img.save(output_path)
    print(f"结果保存至: {output_path}")


# 主预测函数
def predict(input_dir, model_path, output_dir, width=1024, height=768):
    # 设置全局分辨率
    global WIDTH, HEIGHT
    WIDTH = width
    HEIGHT = height
    print(f"预测分辨率: {WIDTH}x{HEIGHT}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    generator = load_generator(model_path, device)

    # 处理所有NEF文件
    nef_files = glob.glob(os.path.join(input_dir, "*.nef"))
    print(f"找到 {len(nef_files)} 个NEF文件")

    for nef_path in nef_files:
        process_nef_file(nef_path, generator, device, output_dir)

    print("预测完成!")


if __name__ == "__main__":
    # 配置参数
    INPUT_DIR = "./images/first"  # 包含NEF文件的目录
    MODEL_PATH = "./models/generator_1024x768_epoch_9.pth"  # 训练好的生成器模型
    OUTPUT_DIR = "./output"  # 输出目录

    predict(INPUT_DIR, MODEL_PATH, OUTPUT_DIR, width=1024, height=768)