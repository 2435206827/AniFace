# AniFace

### 介绍
A face swapping project based on FaceDancer

### 安装教程

1.  `!cd AniFace`
2.  `!pip install . -e requirement.txt`

### 清洗数据集

1.  使用 `!python .\tools\wash.py --dataset_path "your\dataset\path" --output "path\of\result.txt"` ，将在数据集下每一个子目录中随机抽取一张图片，并连续询问是否保留这个子目录，若选择是，则会将该子目录加入result.txt

### 说明

developing