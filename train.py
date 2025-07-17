import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import rawpy
import numpy as np
from PIL import Image
import glob
import time

# 全局分辨率变量
SIZE = (1024, 768)  # (宽度, 高度)

# 创建结果目录
result_dir = "images/result_epoch"
os.makedirs(result_dir, exist_ok=True)


# 数据准备与映射函数
def create_file_mapping(first_dir, end_dir):
    print(f"正在扫描目录: {first_dir} 和 {end_dir}")

    # 获取所有.nef文件 (不区分大小写)
    first_files = []
    for ext in ['.nef', '.NEF']:
        first_files.extend(glob.glob(os.path.join(first_dir, f"*{ext}")))

    # 获取所有.jpg文件 (不区分大小写)
    end_files = []
    for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
        end_files.extend(glob.glob(os.path.join(end_dir, f"*{ext}")))

    print(f"找到 {len(first_files)} 个原始图像文件")
    print(f"找到 {len(end_files)} 个目标图像文件")

    # 创建映射字典
    mapping = {}

    # 根据文件名中的数字序号建立映射
    for f in first_files:
        base = os.path.basename(f)
        # 尝试多种可能的文件名格式
        if 'DSC_' in base:
            # 格式: DSC_5171.nef
            try:
                num_part = base.split('_')[1].split('.')[0]
                num = int(num_part)
            except (IndexError, ValueError):
                print(f"警告: 无法从文件名 '{base}' 中提取数字, 跳过")
                continue
        elif 'DSC' in base:
            # 格式: DSC5171.nef
            try:
                num_part = base.replace('DSC', '').split('.')[0]
                num = int(num_part)
            except ValueError:
                print(f"警告: 无法从文件名 '{base}' 中提取数字, 跳过")
                continue
        else:
            # 尝试从文件名中提取任何数字
            try:
                num_part = ''.join(filter(str.isdigit, base.split('.')[0]))
                num = int(num_part)
            except ValueError:
                print(f"警告: 无法从文件名 '{base}' 中提取数字, 跳过")
                continue

        # 计算目标文件名
        target_num = 100030 + (num - 5171)

        # 尝试多种可能的扩展名
        target_found = False
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            target_path = os.path.join(end_dir, f"{target_num}{ext}")
            if os.path.exists(target_path):
                mapping[f] = target_path
                target_found = True
                break

        if not target_found:
            print(f"警告: 未找到文件 {base} 的目标图像 ({target_num}.jpg)")

    print(f"成功创建 {len(mapping)} 对图像映射")
    return mapping


# 自定义数据集类
class NEFtoJPGDataset(Dataset):
    def __init__(self, mapping, transform=None):
        super().__init__()
        self.mapping = mapping
        self.file_pairs = list(mapping.items())
        self.transform = transform

        # 打印一些样本以验证
        print("\n数据集样本:")
        for i, (nef, jpg) in enumerate(self.file_pairs[:min(3, len(self.file_pairs))]):
            print(f"  {i + 1}. {os.path.basename(nef)} -> {os.path.basename(jpg)}")
        if len(self.file_pairs) > 3:
            print(f"  还有 {len(self.file_pairs) - 3} 对图像...")

    def __getitem__(self, idx):
        nef_path, jpg_path = self.file_pairs[idx]

        try:
            with rawpy.imread(nef_path) as raw:
                rgb = raw.postprocess()
            input_img = Image.fromarray(rgb).resize(SIZE)
        except Exception as e:
            print(f"错误处理 {nef_path}: {e}")
            input_img = Image.new('RGB', SIZE, (0, 0, 0))

        try:
            target_img = Image.open(jpg_path).convert('RGB').resize(SIZE)
        except Exception as e:
            print(f"错误处理 {jpg_path}: {e}")
            target_img = Image.new('RGB', SIZE, (0, 0, 0))

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img

    def __len__(self):
        return len(self.file_pairs)


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


# 简化版判别器
class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 1, 4, padding=1)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat([img_A, img_B], 1)
        return self.model(img_input)


def save_checkpoint(epoch, gen, disc, g_opt, d_opt, path):
    """保存训练检查点（模型+优化器+epoch）"""
    torch.save({
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'g_opt_state_dict': g_opt.state_dict(),
        'd_opt_state_dict': d_opt.state_dict(),
    }, path)
    print(f"检查点已保存: {path}")


def load_checkpoint(path, gen, disc, g_opt, d_opt, device):
    """加载训练检查点"""
    if not os.path.exists(path):
        print(f"检查点文件不存在: {path}")
        return 0

    checkpoint = torch.load(path, map_location=device)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    disc.load_state_dict(checkpoint['disc_state_dict'])
    g_opt.load_state_dict(checkpoint['g_opt_state_dict'])
    d_opt.load_state_dict(checkpoint['d_opt_state_dict'])
    epoch = checkpoint['epoch']

    print(f"成功加载检查点: {path}")
    print(f"从第 {epoch + 1} 轮继续训练")
    return epoch + 1  # 返回下一轮开始的epoch


# 训练函数（支持从检查点恢复）
def train(dataloader, gen, disc, g_opt, d_opt, criterion, device,
          total_epochs=10, start_epoch=0, checkpoint_dir="checkpoints", save_interval=1):
    """训练函数，支持从指定轮数开始"""
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 记录开始时间
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        print(f"\n{'=' * 40}")
        print(f"开始训练第 {epoch + 1}/{total_epochs} 轮")
        print(f"{'=' * 40}")

        for i, (input_imgs, target_imgs) in enumerate(dataloader):
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)

            # 训练判别器
            d_opt.zero_grad()

            # 真实图像
            pred_real = disc(input_imgs, target_imgs)
            loss_real = criterion(pred_real, torch.ones_like(pred_real))

            # 生成图像
            fake_B = gen(input_imgs)
            pred_fake = disc(input_imgs, fake_B.detach())
            loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

            d_loss = (loss_real + loss_fake) * 0.5
            d_loss.backward()
            d_opt.step()

            # 训练生成器
            g_opt.zero_grad()

            pred_fake = disc(input_imgs, fake_B)
            g_adv_loss = criterion(pred_fake, torch.ones_like(pred_fake))

            g_l1_loss = torch.nn.functional.l1_loss(fake_B, target_imgs) * 100

            g_loss = g_adv_loss + g_l1_loss
            g_loss.backward()
            g_opt.step()

            # 每批次打印一次
            batch_time = time.time() - epoch_start
            print(f"[Epoch {epoch + 1}/{total_epochs}] [Batch {i + 1}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                  f"[耗时: {batch_time:.1f}s]")

        # 保存示例
        save_example(gen, input_imgs, target_imgs, epoch, device)

        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(epoch + 1, gen, disc, g_opt, d_opt, checkpoint_path)

            # 同时保存单独的生成器模型（用于预测）
            gen_path = os.path.join(checkpoint_dir, f"generator_epoch_{epoch + 1}.pth")
            torch.save(gen.state_dict(), gen_path)
            print(f"生成器模型已保存: {gen_path}")

        epoch_time = time.time() - epoch_start
        print(f"\n第 {epoch + 1} 轮完成, 耗时: {epoch_time / 60:.2f}分钟")

    total_time = time.time() - start_time
    print(f"\n{'=' * 40}")
    print(f"训练完成! 总耗时: {total_time / 60:.2f}分钟")
    print(f"训练轮数: {start_epoch} -> {total_epochs}")
    print(f"{'=' * 40}")


def save_example(gen, input_imgs, target_imgs, epoch, device):
    gen.eval()
    with torch.no_grad():
        # 只处理第一个样本
        fake_imgs = gen(input_imgs[:1])

        # 创建保存目录
        save_dir = os.path.join(result_dir, f"epoch_{epoch + 1}")
        os.makedirs(save_dir, exist_ok=True)

        # 保存输入图像
        input_img = input_imgs[0].cpu().numpy().transpose(1, 2, 0)
        input_img = (input_img * 0.5 + 0.5) * 255
        input_img = np.clip(input_img, 0, 255).astype(np.uint8)
        Image.fromarray(input_img).save(os.path.join(save_dir, "input.jpg"))

        # 保存生成图像
        fake_img = fake_imgs[0].cpu().numpy().transpose(1, 2, 0)
        fake_img = (fake_img * 0.5 + 0.5) * 255
        fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
        Image.fromarray(fake_img).save(os.path.join(save_dir, "generated.jpg"))

        # 保存目标图像
        target_img = target_imgs[0].cpu().numpy().transpose(1, 2, 0)
        target_img = (target_img * 0.5 + 0.5) * 255
        target_img = np.clip(target_img, 0, 255).astype(np.uint8)
        Image.fromarray(target_img).save(os.path.join(save_dir, "target.jpg"))

        print(f"示例图像保存到: {save_dir}")

    gen.train()


# 主训练函数（支持继续训练）
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # ========== 配置参数 ========== #
    # 原始图像目录
    first_dir = "./images/first"
    # 目标图像目录
    end_dir = "./images/end"
    # 总训练轮数
    total_epochs = 15
    # 批次大小
    batch_size = 1
    # 学习率
    lr = 2e-4
    # 检查点目录
    checkpoint_dir = "checkpoints"
    # 保存间隔（每几轮保存一次）
    save_interval = 3

    # 恢复训练设置
    resume_training = True  # 是否从检查点恢复训练
    # 指定要恢复的检查点路径（如果为None，则自动使用最新的检查点）
    resume_checkpoint = None
    # ============================= #

    print("\n" + "=" * 50)
    print(f" 开始图像转换训练")
    print(f" 分辨率: {SIZE[0]}x{SIZE[1]}")
    print(f" 总训练轮数: {total_epochs}")
    print(f" 恢复训练: {'是' if resume_training else '否'}")
    print("=" * 50 + "\n")

    print(f"原始图像目录: {first_dir}")
    print(f"目标图像目录: {end_dir}")

    # 创建文件映射
    mapping = create_file_mapping(first_dir, end_dir)
    if not mapping:
        print("\n错误: 没有找到匹配的图像对，请检查以下内容:")
        print("1. 确保目录路径正确")
        print("2. 确保原始图像是.nef格式")
        print("3. 确保目标图像是.jpg格式")
        print("4. 文件命名符合DSC_XXXX模式")
        print("\n正在退出...")
        return

    print(f"\n成功匹配 {len(mapping)} 对训练图像")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 创建数据集
    dataset = NEFtoJPGDataset(mapping, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    gen = SimpleGenerator().to(device)
    disc = SimpleDiscriminator().to(device)

    # 打印模型信息
    gen_params = sum(p.numel() for p in gen.parameters())
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"\n生成器参数数量: {gen_params:,}")
    print(f"判别器参数数量: {disc_params:,}")

    # 优化器和损失函数
    g_opt = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # 起始轮数
    start_epoch = 0

    # 恢复训练
    if resume_training:
        # 自动查找最新的检查点
        if resume_checkpoint is None:
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
            if checkpoints:
                # 按epoch排序
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                resume_checkpoint = checkpoints[-1]

        if resume_checkpoint:
            start_epoch = load_checkpoint(resume_checkpoint, gen, disc, g_opt, d_opt, device)

    print(f"\n开始训练，从第 {start_epoch} 轮到第 {total_epochs} 轮...")

    # 开始训练
    train(
        dataloader=dataloader,
        gen=gen,
        disc=disc,
        g_opt=g_opt,
        d_opt=d_opt,
        criterion=criterion,
        device=device,
        total_epochs=total_epochs,
        start_epoch=start_epoch,
        checkpoint_dir=checkpoint_dir,
        save_interval=save_interval
    )

    print("\n训练完成! 模型和示例图像已保存")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    main()
