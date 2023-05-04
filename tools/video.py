import ffmpeg
import argparse
import os
from form import form

def video2images(file_path, interval = 5, output_path = ""):
    (
        ffmpeg
        .input(file_path)
        .filter("fps", fps = 1 / interval)
        .output("{}/frame%06d.png".format(output_path))
        .run()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("-p", "--video_path", type = str, required = True, help = "path of your video or video dir. use -b or --batch to proccess videos in batch")
    parser.add_argument("-o", "--output_path", type = str, required = True, help = "output directory")
    parser.add_argument("-i", "--interval", type = int, default = 5, help = "interval seconds for frames")
    parser.add_argument("-b", "--batch", action = "store_false", help = "batch proccessing")

    args = parser.parse_args().__dict__

    print(form("<yellow_ex>Starting Process"))

    if not args["batch"]:
        # solo
        try:
            video2images(args["video_path"], args["interval"], args["output_path"])
        except Exception as e:
            print(form("<red_ex>" + str(e)))
        print(form("<green_ex>Process completed"))

    else:
        # batch
        assert os.path.exists(args["video_path"]), "{} is not exists".format(args["video_path"])
        
        tot = 0
        for subdir, _, files in os.walk(args["video_path"]):
            for file in files:
                if file.endswith('.mp4') or file.endswith('.avi') or file.endswith('.flv'):
                    try:
                        path = os.path.join(subdir, file)
                        dir = ".".join(str(os.path.join(subdir, file)).split(".")[: -1])
                        os.mkdir(dir)
                        video2images(path, args["interval"], dir)
                        print(form("<green>{}th video".format(tot)))
                    except Exception as e:
                        print(form("<red_ex>" + str(e)))
                    tot += 1
        print(form("<green_ex>Process {} videos completed".format(tot)))
