import os
import random
import tkinter as tk
from tkinter.simpledialog import Dialog
from tkinter import Tk, Label
from PIL import ImageTk, Image
from alive_progress import alive_bar

root = Tk()

class ImageDialog(Dialog):
    """
    WARNING: Don't minimize the window, otherwise it will not open again
    """
    def __init__(self, parent, title, image):
        self.image = image
        super().__init__(parent, title)

    def body(self, master):
        Label(master, image=self.image).pack()
        return None
    
    def apply(self):
        self.result = True

def wash(dir_path):
    """
    chosing the proper image of faces from your dataset
    """
    selected_dirs = []
    lens = len([dirs for dirs, _, _ in os.walk(dir_path)])
    with alive_bar(lens, title = "data processing") as bar:
        for subdir, _, _ in os.walk(dir_path):
            image_files = [f for f in os.listdir(subdir) if f.endswith('.jpg') or f.endswith('.png')]
            if image_files:
                selected_image = random.choice(image_files)
                img = ImageTk.PhotoImage(Image.open(os.path.join(subdir, selected_image)).resize((128, 128)))
                # show image
                dialog = ImageDialog(root, 'is it a face?', img)
                if dialog.result:
                    selected_dirs.append(subdir)
                bar()

    print("Processing Successful")
    return selected_dirs

s = wash("D:\\icf\\icartoonface_rectrain_02000")
root.mainloop()
f = open("D:\\icf\\2000.txt", "w")
f.write("\n".join(s))