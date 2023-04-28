import os
import argparse
import itertools
from PIL import Image
from alive_progress import alive_bar
from colorama import Fore
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")

def ratio_process(directory, ratio, quiet):

    transform_pad = transforms.Compose([
        transforms.ToTensor(),
        SquarePad(),
        transforms.Resize((256, 256)),
        transforms.ToPILImage()
    ])

    transform_resize = transforms.Compose([
        transforms.Resize((256, 256))
    ])
    
    lens = len(list(itertools.chain.from_iterable([f for f in [files for _, _, files in os.walk(directory)]]))) - 1
    with alive_bar(lens, title = Fore.WHITE + "data processing") as bar:
        for subdir, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    try:
                        image_path = os.path.join(subdir, file)
                        image = Image.open(image_path)
                        width, height = image.size
                        aspect_ratio = max(width / height, height / width)
                        if aspect_ratio <= ratio:
                            image = transform_resize(image)
                        else:
                            image = transform_pad(image)
                        image.save(image_path)
                    except Exception as e:
                        pass
                        # if not quiet:
                        # # print(e)
                        #     print(Fore.LIGHTRED_EX + "Error occuring when process " + Fore.RED + str(file) + Fore.LIGHTRED_EX + ", deleting files.")
                bar()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--dataset_path", type = str, required = True, help = "path of your dataset")
    parser.add_argument("--ratio", type = float, default = 1.1, help = "The aspect ratio threshold, above which the image will no longer be scaled")
    parser.add_argument("--quiet", type = bool, default = True, help = "echo quiet")

    args = parser.parse_args().__dict__

    s = ratio_process(args["dataset_path"], args["ratio"], args["quiet"])