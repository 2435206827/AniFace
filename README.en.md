# AniFace

### Description
A FaceSwap experimental project based on FaceDancer architecture suitable for anime faces

### Installation

1.  `!cd AniFace`
2.  `!pip install . -e requirement.txt` (Not Built)

### Dataset Cleaning
1.  `!python .\tools\select.py`
    - Randomly select one image from each subdirectory in the dataset and continuously ask if to keep this subdirectory. If you choose Yes, the subdirectory will be added to result.txt
    - e.g. `!python .\tools\select.py --dataset_path "your\dataset\path" --output "path\of\result.txt`
2. `!python .\tools\ratio.py`
    - Fill and trim all images in the dataset to a uniform size, and detect images which are not in RGB (3 channels)
    - e.g. `!python .\tools\ratio.py --dataset_path "your\dataset\path" --size 256 --ratio 1.1 --del`
3. `!python .\tools\less.py`
    - Delete sub directories with few images
    - e.g. `!python .\tools\less.py --dataset_path "your\dataset\path" --num 114514`

### ArcFace Training
Train ArcFace through `./arcface_training` 

If it already exists, you can skip this step

1. Configure ArcFace training parameters through C
    - `train_root` is the dataset directory
    - `checkpoints_path` is the location where the model to be saved
2. Use `./arcface_training/train.py` for training

### Prepare pretrained models

<!-- 1. Place the pretrained ResNet in `./model/ResNet` -->
1. Place the pre trained ArcFace in `./model/arcface`
    - You can use this project for training (default to ResNet-50 backbone)

### Instructions

developing
