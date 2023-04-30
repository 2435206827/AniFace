import sys
sys.path.append("..")

import os
import argparse
import shutil
from utils.form import form

def delete_less(root_dir, n):
    """
    delete dirs with few images
    """
    tot = 0
    for dirpath, _, filenames in os.walk(root_dir):
        image_count = 0
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_count += 1
        if image_count < n and dirpath != root_dir:
            shutil.rmtree(dirpath)
            tot += 1
    return tot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "delete dirs with few images")
    parser.add_argument("-p", "--dataset_path", type = str, required = True, help = "path of your dataset")
    parser.add_argument("-n", "--num", type = int, default = 1, help = "Directories with fewer than num images will be deleted")

    args = parser.parse_args().__dict__

    assert not args["num"] < 1, "illegal number"

    tot = delete_less(args["dataset_path"], args["num"])
    # BUG: 无法格式化颜色输出
    print(form("<green_ex>Successfully Completed. <red>{_tot}<green_ex> dirs have been deleted".format(_tot = tot)))