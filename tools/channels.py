import sys
sys.path.append(".")
sys.path.append("..")

from PIL import Image
import os
import argparse
from utils.form import form

def remove_non_rgb_images(directory):
    tot = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                if image.mode != "RGB":
                    os.remove(image_path)
                    tot += 1
                    # print(image_path)
    return tot
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "delete images not in RGB (3 channels)")
    parser.add_argument("-p", "--dataset_path", type = str, required = True, help = "path of your dataset")

    args = parser.parse_args().__dict__

    tot = remove_non_rgb_images(args["dataset_path"])
    print(form("<green_ex>Successfully Completed. <red>{_tot}<green_ex> imgs have been deleted".format(_tot = tot)))