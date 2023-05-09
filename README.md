# AniFace

### 介绍
一个架构基于FaceDancer，适用于动漫人脸的FaceSwap实验项目

### 安装教程

1.  `!cd AniFace`
2.  `!pip install . -e requirement.txt` （未构建）

### 清洗数据集

1.  `!python .\tools\select.py`
    - 将在数据集下每一个子目录中随机抽取一张图片，并连续询问是否保留这个子目录，若选择是，则会将该子目录加入result.txt
    - 示例 `!python .\tools\select.py --dataset_path "your\dataset\path" --output "path\of\result.txt"`
2. `!python .\tools\ratio.py`
    - 将数据集所有图片填充并修剪为统一的大小，并检测通道不为 `RGB` (3) 的图片
    - 示例 `!python .\tools\ratio.py --dataset_path "your\dataset\path" --size 256 --ratio 1.1 --del`
3. `!python .\tools\less.py`
    - 删除图像数较少的子目录
    - 示例 `!python .\tools\less.py --dataset_path "your\dataset\path" --num 114514`

### ArcFace训练
通过 `./arcface_training` 训练ArcFace

如果已有，可以跳过此步

1. 通过 `./arcface_training/config.yml` 配置ArcFace训练参数
    - `train_root` 数据集目录
    - `checkpoints_path` 指定模型保存位置
2. 使用 `./arcface_training/train.py` 进行训练

### 准备预训练模型

<!-- 1. 将预训练的ResNet放到 `./model/ResNet` -->
1. 将预训练的ArcFace放到 `./model/arcface` 


### 说明

developing
