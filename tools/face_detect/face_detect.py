import sys
sys.path.append(".")
import cv2
import os
import argparse
import shutil
from tools.form import *

def face_detect(img_path, output_path, test = False, cascade_name = "D:/Users/program/Anaconda3/envs/oldpy/Lib/site-packages/cv2/data/lbpcascade_animeface.xml"):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)  # normalizing hist.
    face_cascade = cv2.CascadeClassifier(cascade_name)
    # face_cascade.load("D:/Users/program/Anaconda3/envs/oldpy/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img_gray)  # detect
    if not test:
        for (x, y, w, h) in faces:
            print("1")
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 5)
        cv2.imwrite(os.path.join(output_path, img_path), img)
        cv2.imshow("Face detection", cv2.resize(img, (256, 256)))
        cv2.waitKey(0)
    else:
        for (x, y, w, h) in faces:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Face detection by OpenCV, the output will be faces.")
    parser.add_argument("-p", "--imgs_path", type = str, required = True, help = "path of imgs or imgs dir. use -b or --batch to proccess imgs in batches")
    parser.add_argument("-o", "--output_path", type = str, required = True, help = "output directory")
    parser.add_argument("-b", "--batch", action = "store_false", help = "batch proccessing")
    parser.add_argument("-v", "--visualization", action = "store_false", help = "for testing")

    args = parser.parse_args().__dict__


    print(form("<yellow_ex>Starting Process"))

    if args["batch"]:
        # solo
        try:
            print(os.path.basename(args["imgs_path"]))
            print(dir)
            face_detect(os.path.basename(args["imgs_path"]), args["output_path"])
        except Exception as e:
            print(form("<red_ex>" + str(e)))
        print(form("<green_ex>Process completed"))

    else:
        # batch
        assert os.path.exists(args["imgs_path"]), "{} is not exists".format(args["imgs_path"])
        
        tot = 0
        for subdir, _, files in os.walk(args["imgs_path"]):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".bmp"):
                    # try:
                    path = os.path.join(subdir, file)
                    dir = os.path.join(args["output_path"], os.path.basename(subdir))
                    if not os.path.exists(dir): os.mkdir(dir)
                    print(path)
                    print(dir)
                    face_detect(path, dir)
                    # except Exception as e:
                    #     print(form("<red_ex>" + str(e)))
                    tot += 1
        print(form("<green_ex>Detected {} images completed".format(tot)))
