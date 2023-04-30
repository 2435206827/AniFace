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
    - 将数据集所有图片填充并修剪为统一的大小
    - 示例 `!python .\tools\ratio.py --dataset_path "your\dataset\path" --size 256 --ratio 1.1 --del`
3. `!python .\tools\less.py`
    - 删除图像数较少的子目录
    - 示例 `!python .\tools\less.py --dataset_path "your\dataset\path" --num 114514`

### 说明

developing