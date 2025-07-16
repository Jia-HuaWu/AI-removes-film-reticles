# 使用AI去除胶片色罩
本项目基于 “图生图” 的算法思想，使用`pytorch`及其相关库实现：将 `.nef` 格式的胶片原片处理成 `.jpg` 格式的去除色罩后的图片。

---

## 怎么使用
运行 `RunModel.py` 文件，可以把 `./new_images` 目录下的 `.NEF` 格式图片转换成 `.jpg` 格式，并保存在 `./result` 目录下

在 `RunModel.py` 文件的末尾，可以看见这么一段代码：
```
if __name__ == "__main__":
    # 配置参数
    INPUT_DIR = "./new_images"  # 包含NEF文件的目录
    MODEL_PATH = "./models/generator_1024x768_epoch_9.pth"  # 训练好的生成器模型
    OUTPUT_DIR = "./result"  # 输出目录

    predict(INPUT_DIR, MODEL_PATH, OUTPUT_DIR, width=1024, height=768)
```
这里可以修改你的输入、输出目录，选择哪个训练模型（这里使用的是我训练的一个分辨率为1024x768的模型），以及输出时的分辨率（建议和训练时的分辨率一致）。

模型的命名规则是 `generator_分辨率_epoch_训练轮数.pth`

---

## 怎么训练自己的模型
`main.py` 文件是用于基于你的数据来训练模型的，运行这个代码就可以用你自己的数据来训练模型。
运行代码时，使用的训练数据来自 `./images/first` 目录（未处理的.nef格式的图片）和 `./images/end` （处理后的.jpg格式图片）
**值得注意的是：**.nef格式图片请以 `DSC_5171.nef`、`DSC_5172.nef`、`DSC_5173.nef`......的形式来命名。而.jpg格式图片请以 `100030.jpg`、`100031.jpg`、`100032.jpg`......的形式来命名。而且要保证 `DSC_xxx.nef` 去除色罩后的图片是 `100030 + xxx - 5171.jpg`
(因为我拿到的数据是这样的，懒得改了X) )

写代码时加了很多断点来查bug，写完后不仅没删还变本加厉的加了很多的断点。

其实是特意保留了很多断点，让你知道这个代码在运行()。

代码最上方
```
WIDTH = 1024
HEIGHT = 768
print(f"[初始化] 设置分辨率: {WIDTH}x{HEIGHT}")
```
可以设置成自己想要的分辨率，设备不是很好的话不建议设置太高的分辨率。

这两个变量用于设置输入数据位置
`first_dir`是原始数据(.NEF)
`end_dir`是处理后的数据(.jpg)
```
first_dir = "./images/first"
end_dir = "./images/end"
```

这两个是学习率和训练轮数
不知道学习率是啥的可以不用动
```
lr = 2e-4 #学习率
epochs = 10 #训练轮数
```

---

## 先写到这，后面想到再说