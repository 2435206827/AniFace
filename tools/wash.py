import sys
sys.path.append("..")

import os
import random
import argparse
from utils.form import form
from tkinter.simpledialog import Dialog
from tkinter import Tk, Label
from PIL import ImageTk, Image
from alive_progress import alive_bar

class ImageDialog(Dialog):
    """
    WARNING: Don't minimize the window, otherwise it will not open again
    """
    def __init__(self, parent, title, image):
        self.image = image
        super().__init__(parent, title)

    def body(self, master):
        Label(master, image = self.image).pack()
        return None
    
    def apply(self):
        self.result = True

def wash(dir_path):
    """
    chosing the proper image of faces from your dataset
    """
    selected_dirs = []
    lens = len([dirs for dirs, _, _ in os.walk(dir_path)]) - 1
    with alive_bar(lens, title = "data processing") as bar:
        for subdir, _, _ in os.walk(dir_path):
            image_files = [f for f in os.listdir(subdir) if f.endswith('.jpg') or f.endswith('.png')]
            if image_files:
                selected_image = random.choice(image_files)
                img = ImageTk.PhotoImage(Image.open(os.path.join(subdir, selected_image)).resize((128, 128)))
                # show image
                dialog = ImageDialog(root, 'is it a human face?', img)
                if dialog.result:
                    selected_dirs.append(subdir)
                bar()

    return selected_dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Simple tool to quickly select the image subdirectories required for the dataset.")
    parser.add_argument("--dataset_path", type = str, required = True, help = "path of your dataset")
    parser.add_argument("--output", type = str, required = True, help = "file path of the result")

    args = parser.parse_args().__dict__

    print(form("<yellow_ex>Starting Process"))
    root = Tk()
    s = wash(args["dataset_path"])
    print(form("<green_ex>Successful Completed"))
    root.mainloop()

    f = open(args["output"], "w")
    f.write("\n".join(s))