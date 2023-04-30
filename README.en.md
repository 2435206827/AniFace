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
    - Fill and trim all images in the dataset to a uniform size
    - e.g. `!python .\tools\ratio.py --dataset_path "your\dataset\path" --size 256 --ratio 1.1 --del`

### Instructions

developing