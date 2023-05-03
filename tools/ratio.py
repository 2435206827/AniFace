import sys
sys.path.append(".")
sys.path.append("..")

import os
import argparse
import itertools
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from utils.form import form

def _pad_calc(H, W):
    # There is no need to handle the parity issue of filling pixels, it will be resolved in Resize
    p = abs(H - W) // 2
    if H > W:
        return (p, 0, p, 0)
    if H < W:
        return (0, p, 0, p)
    else:
        return (0, 0, 0, 0)

def ratio_process(directory, size, ratio, delete):
    lens = len(list(itertools.chain.from_iterable([f for f in [files for _, _, files in os.walk(directory)]])))
    tot, errs, rm_list = 0, 0, []
    with tqdm(total = lens, desc = "data processing", leave = True) as bar:
        for subdir, _, files in os.walk(directory):
            for file in files:
                
                if file.endswith('.jpg') or file.endswith('.png'):
                    tot += 1
                    try:
                        image_path = os.path.join(subdir, file)
                        image = Image.open(image_path)

                        if image.mode != "RGB":
                            raise Exception("{} is not in channels of RGB".format(tot))
                        
                        width, height = image.size
                        aspect_ratio = max(width / height, height / width)
                        if aspect_ratio <= ratio:
                            image = transforms.Compose([
                                transforms.Pad(_pad_calc(height, width)),
                                transforms.Resize((size, size))
                            ])(image)
                        else:
                            image = transforms.Resize((size, size))(image)
                        image.save(image_path)

                    except Exception as e:
                        errs += 1
                        print(e)
                        rm_list.append(image_path) if delete else None
                bar.update()

    for p in rm_list:
        os.remove(p)
    print(form("<red_ex>{0} files have been removed".format(len(rm_list)))) if delete else None
    return tot, errs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Fill & trim all images in the dataset to a uniform size, and detect images which are not in RGB (3 channels)")
    parser.add_argument("-p", "--dataset_path", type = str, required = True, help = "path of your dataset")
    parser.add_argument("-r", "--ratio", type = float, default = 1.1, help = "The aspect ratio threshold >= 1, above which the image will no longer be scaled")
    parser.add_argument("-s", "--size", type = int, default = 256, help = "The w&h for output")
    parser.add_argument("-d", "--del", action = "store_true", help = "Delete images which are failed to been process")
    # parser.add_argument("-q", "--quiet", action = "store_false", help = "echo quiet")

    args = parser.parse_args().__dict__

    assert not args["ratio"] < 1, "illegal ratio"
    assert not args["size"] < 1, "illegal image size"

    print(form("<yellow_ex>Starting Process"))
    tot, errs = ratio_process(
        args["dataset_path"], 
        args["size"],
        args["ratio"], 
        args["del"]
    )
    print(form("<green_ex>Successfully Completed. {_tot} in total, <red>{_errs}<green> errors occured".format(_errs = errs, _tot = tot)))