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
import matplotlib.pyplot as plt

# 全局分辨率变量 - 修改为支持宽高
WIDTH = 1024
HEIGHT = 768
print(f"[初始化] 设置分辨率: {WIDTH}x{HEIGHT}")


# 2. 数据准备与映射函数
def create_file_mapping(first_dir, end_dir):
    print(f"[文件映射] 开始创建文件映射: {first_dir} -> {end_dir}")
    mapping = {}
    first_files = sorted(glob.glob(os.path.join(first_dir, "*.nef")))
    end_files = sorted(glob.glob(os.path.join(end_dir, "*.jpg")))
    print(f"[文件映射] 找到 {len(first_files)} 个NEF文件, {len(end_files)} 个JPG文件")

    for f in first_files:
        base = os.path.basename(f)
        num = int(base.split('_')[1].split('.')[0])
        target_num = 100030 + (num - 5171)
        target_path = os.path.join(end_dir, f"{target_num}.jpg")
        if os.path.exists(target_path):
            mapping[f] = target_path
            print(f"[文件映射] 匹配: {os.path.basename(f)} -> {os.path.basename(target_path)}")
        else:
            print(f"[警告] 找不到匹配文件: {os.path.basename(f)} 的目标 {target_path}")

    print(f"[文件映射] 完成! 共找到 {len(mapping)} 对有效图像")
    return mapping


# 3. 自定义数据集类
class NEFtoJPGDataset(Dataset):
    def __init__(self, mapping, transform=None):
        self.mapping = mapping
        self.file_pairs = list(mapping.items())
        self.transform = transform
        print(f"[数据集] 初始化数据集, 共有 {len(self.file_pairs)} 对图像")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        nef_path, jpg_path = self.file_pairs[idx]
        print(f"[数据集] 加载图像对 #{idx}: {os.path.basename(nef_path)} -> {os.path.basename(jpg_path)}")

        try:
            with rawpy.imread(nef_path) as raw:
                rgb = raw.postprocess()
            input_img = Image.fromarray(rgb)
            print(f"[数据集] 成功加载NEF: {os.path.basename(nef_path)}, 尺寸: {input_img.size}")
        except Exception as e:
            print(f"[错误] 处理NEF失败 {nef_path}: {e}")
            input_img = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))

        try:
            target_img = Image.open(jpg_path).convert('RGB')
            print(f"[数据集] 成功加载JPG: {os.path.basename(jpg_path)}, 尺寸: {target_img.size}")
        except Exception as e:
            print(f"[错误] 处理JPG失败 {jpg_path}: {e}")
            target_img = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))

        if self.transform:
            print(f"[数据集] 应用变换...")
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            print(f"[数据集] 变换后尺寸: 输入={input_img.shape}, 目标={target_img.shape}")

        return input_img, target_img


# 4. 定义生成器 (U-Net架构) - 支持自定义分辨率
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        print(f"[生成器] 初始化U-Net生成器, 输入通道={in_channels}, 输出通道={out_channels}")
        print(f"[生成器] 目标分辨率: {WIDTH}x{HEIGHT}")

        def down_block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def up_block(in_ch, out_ch, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return layers

        # 基础结构
        self.down1 = nn.Sequential(*down_block(in_channels, 64, normalize=False))
        self.down2 = nn.Sequential(*down_block(64, 128))
        self.down3 = nn.Sequential(*down_block(128, 256))
        self.down4 = nn.Sequential(*down_block(256, 512))
        print(f"[生成器] 添加基础下采样块: d1-d4")

        # 对于高分辨率图像，添加更多层
        max_dim = max(WIDTH, HEIGHT)
        if max_dim > 512:
            print(f"[生成器] 添加高分辨率扩展 (max_dim={max_dim}>512)")
            self.down5 = nn.Sequential(*down_block(512, 512))
            self.down6 = nn.Sequential(*down_block(512, 512))
            self.up0 = nn.Sequential(*up_block(512, 512, dropout=True))
            self.up1 = nn.Sequential(*up_block(1024, 512, dropout=True))
            self.up2 = nn.Sequential(*up_block(1024, 256))
            self.up3 = nn.Sequential(*up_block(512, 128))
            self.up4 = nn.Sequential(*up_block(256, 64))
            self.final = nn.Sequential(
                nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
                nn.Tanh()
            )
            print(f"[生成器] 添加额外下采样和上采样块: d5-d6, u0-u4")
        else:
            print(f"[生成器] 使用标准分辨率结构 (max_dim={max_dim}<=512)")
            self.up1 = nn.Sequential(*up_block(512, 256, dropout=True))
            self.up2 = nn.Sequential(*up_block(512, 128))
            self.up3 = nn.Sequential(*up_block(256, 64))
            self.final = nn.Sequential(
                nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
                nn.Tanh()
            )

        print(f"[生成器] 初始化完成")

    def forward(self, x):
        print(f"[生成器] 前向传播, 输入尺寸: {x.shape}")

        # 下采样
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        print(f"[生成器] 下采样后: d1={d1.shape}, d2={d2.shape}, d3={d3.shape}, d4={d4.shape}")

        # 高分辨率处理
        max_dim = max(WIDTH, HEIGHT)
        if max_dim > 512:
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            print(f"[生成器] 额外下采样: d5={d5.shape}, d6={d6.shape}")

            u0 = self.up0(d6)
            print(f"[生成器] 上采样 u0: {u0.shape}")

            u1 = self.up1(torch.cat([u0, d5], 1))
            print(f"[生成器] 上采样 u1 (拼接u0+d5): {u1.shape}")

            u2 = self.up2(torch.cat([u1, d4], 1))
            print(f"[生成器] 上采样 u2 (拼接u1+d4): {u2.shape}")

            u3 = self.up3(torch.cat([u2, d3], 1))
            print(f"[生成器] 上采样 u3 (拼接u2+d3): {u3.shape}")

            u4 = self.up4(torch.cat([u3, d2], 1))
            print(f"[生成器] 上采样 u4 (拼接u3+d2): {u4.shape}")

            u5 = self.final(torch.cat([u4, d1], 1))
            print(f"[生成器] 最终输出: {u5.shape}")
            return u5
        else:
            u1 = self.up1(d4)
            u2 = self.up2(torch.cat([u1, d3], 1))
            u3 = self.up3(torch.cat([u2, d2], 1))
            u4 = self.final(torch.cat([u3, d1], 1))
            print(f"[生成器] 上采样路径: u1={u1.shape}, u2={u2.shape}, u3={u3.shape}, 输出={u4.shape}")
            return u4


# 5. 定义判别器 (PatchGAN) - 支持不同分辨率
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        print(f"[判别器] 初始化PatchGAN判别器, 输入通道={in_channels}")
        print(f"[判别器] 目标分辨率: {WIDTH}x{HEIGHT}")

        def block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        # 基础结构
        layers = [
            *block(in_channels * 2, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
        ]
        print(f"[判别器] 添加基础块: 4层")

        # 对于高分辨率图像，添加更多层
        max_dim = max(WIDTH, HEIGHT)
        if max_dim > 512:
            print(f"[判别器] 添加高分辨率扩展 (max_dim={max_dim}>512)")
            layers.extend([
                *block(512, 512),
                *block(512, 512),
            ])
            print(f"[判别器] 添加额外2层")

        # 最终层
        layers.extend([
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        ])
        print(f"[判别器] 添加最终层")

        self.model = nn.Sequential(*layers)
        print(f"[判别器] 初始化完成")

    def forward(self, img_A, img_B):
        print(f"[判别器] 前向传播, 输入A: {img_A.shape}, 输入B: {img_B.shape}")
        img_input = torch.cat([img_A, img_B], 1)
        print(f"[判别器] 拼接后输入: {img_input.shape}")
        output = self.model(img_input)
        print(f"[判别器] 输出: {output.shape}")
        return output


# 6. 训练函数
def train(dataloader, gen, disc, g_opt, d_opt, criterion, device, epochs=100):
    print(f"[训练] 开始训练, 共 {epochs} 个epochs")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    print(f"[训练] 模型保存目录: {model_dir}")

    # 根据分辨率调整打印频率
    max_dim = max(WIDTH, HEIGHT)
    print_freq = 10 if max_dim <= 512 else 5
    print(f"[训练] 日志打印频率: 每 {print_freq} 个batch")

    for epoch in range(epochs):
        print(f"\n[训练] === Epoch {epoch + 1}/{epochs} ===")

        for i, (input_imgs, target_imgs) in enumerate(dataloader):
            print(f"\n[训练] Batch {i + 1}/{len(dataloader)}")

            real_A = input_imgs.to(device)
            real_B = target_imgs.to(device)
            print(f"[训练] 加载到设备: 输入={real_A.shape}, 目标={real_B.shape}")

            # 训练判别器
            print(f"[训练] 训练判别器...")
            d_opt.zero_grad()

            # 真实图像
            pred_real = disc(real_A, real_B)
            loss_real = criterion(pred_real, torch.ones_like(pred_real))
            print(f"[训练] 判别器真实损失: {loss_real.item():.4f}")

            # 生成图像
            with torch.no_grad():
                fake_B = gen(real_A)
                print(f"[训练] 生成图像: {fake_B.shape}")

            pred_fake = disc(real_A, fake_B.detach())
            loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
            print(f"[训练] 判别器生成损失: {loss_fake.item():.4f}")

            d_loss = (loss_real + loss_fake) * 0.5
            d_loss.backward()
            d_opt.step()
            print(f"[训练] 判别器总损失: {d_loss.item():.4f}, 已更新权重")

            # 训练生成器
            print(f"[训练] 训练生成器...")
            g_opt.zero_grad()

            pred_fake = disc(real_A, fake_B)
            g_adv_loss = criterion(pred_fake, torch.ones_like(pred_fake))
            print(f"[训练] 生成器对抗损失: {g_adv_loss.item():.4f}")

            g_l1_loss = torch.nn.L1Loss()(fake_B, real_B) * 100
            print(f"[训练] 生成器L1损失: {g_l1_loss.item():.4f}")

            g_loss = g_adv_loss + g_l1_loss
            g_loss.backward()
            g_opt.step()
            print(f"[训练] 生成器总损失: {g_loss.item():.4f}, 已更新权重")

            # 使用全局变量控制打印频率
            if i % print_freq == 0:
                print(f"[训练] [Epoch {epoch + 1}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # 保存模型
        gen_path = os.path.join(model_dir, f"generator_{WIDTH}x{HEIGHT}_epoch_{epoch}.pth")
        disc_path = os.path.join(model_dir, f"discriminator_{WIDTH}x{HEIGHT}_epoch_{epoch}.pth")

        torch.save(gen.state_dict(), gen_path)
        torch.save(disc.state_dict(), disc_path)
        print(f"[训练] 模型已保存: {gen_path}")

        # 保存示例
        save_example(gen, real_A, real_B, epoch)


def save_example(gen, input_imgs, target_imgs, epoch, max_examples=1):
    print(f"[示例] 保存示例图像, epoch={epoch}")
    gen.eval()
    with torch.no_grad():
        batch_size = input_imgs.size(0)
        num_examples = min(batch_size, max_examples)
        print(f"[示例] 选择 {num_examples} 个示例")

        fake_imgs = gen(input_imgs[:num_examples])
        print(f"[示例] 生成图像完成, 尺寸: {fake_imgs.shape}")

        display_list = [input_imgs[:num_examples], fake_imgs, target_imgs[:num_examples]]
        titles = ['Input', 'Generated', 'Target']

        # 根据分辨率调整图像大小
        figsize = (15, 5 * num_examples) if max(WIDTH, HEIGHT) <= 512 else (20, 7 * num_examples)
        fig, axes = plt.subplots(num_examples, 3, figsize=figsize)
        print(f"[示例] 创建图像网格: {num_examples}x3, 尺寸={figsize}")

        if num_examples == 1:
            axes = axes[np.newaxis, :]

        for i in range(num_examples):
            for j in range(3):
                ax = axes[i, j]
                img = display_list[j][i].cpu().permute(1, 2, 0).numpy()
                img = (img * 0.5 + 0.5)
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                ax.set_title(f"{titles[j]} - {WIDTH}x{HEIGHT}")
                ax.axis('off')

        plt.tight_layout()
        save_path = f"results_{WIDTH}x{HEIGHT}_epoch_{epoch}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"[示例] 示例图像已保存: {save_path}")
    gen.train()


# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[主函数] 使用设备: {device}")
    print(f"[主函数] 目标分辨率: {WIDTH}x{HEIGHT}")

    # 根据分辨率调整批次大小
    max_dim = max(WIDTH, HEIGHT)
    if max_dim <= 256:
        batch_size = 4
    elif max_dim <= 512:
        batch_size = 2
    else:  # 高分辨率
        batch_size = 1
    print(f"[主函数] 批次大小: {batch_size} (基于最大维度 {max_dim})")

    lr = 2e-4
    epochs = 10
    print(f"[主函数] 学习率: {lr}, Epochs: {epochs}")

    first_dir = "./images/first"
    end_dir = "./images/end"
    print(f"[主函数] 原始图像目录: {first_dir}")
    print(f"[主函数] 目标图像目录: {end_dir}")

    mapping = create_file_mapping(first_dir, end_dir)
    if not mapping:
        print("[错误] 没有找到匹配的图像对, 退出程序")
        return

    # 使用全局分辨率变量
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),  # 高度, 宽度
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    print(f"[主函数] 数据转换: Resize({HEIGHT}, {WIDTH}), ToTensor, Normalize")

    dataset = NEFtoJPGDataset(mapping, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[主函数] 创建数据加载器, 批次大小={batch_size}, 数据长度={len(dataset)}")

    gen = UNetGenerator().to(device)
    disc = Discriminator().to(device)
    print(f"[主函数] 模型已移动到 {device}")

    # 打印模型信息
    print(f"[主函数] 生成器参数数量: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"[主函数] 判别器参数数量: {sum(p.numel() for p in disc.parameters()):,}")

    g_opt = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    print(f"[主函数] 优化器和损失函数已初始化")

    # 创建目录
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print(f"[主函数] 输出目录已创建")

    print(f"[主函数] 开始训练...")
    train(dataloader, gen, disc, g_opt, d_opt, criterion, device, epochs)
    print(f"[主函数] 训练完成!")


if __name__ == "__main__":
    print("===== 程序开始 =====")
    main()
    print("===== 程序结束 =====")