"""
Generate videos from episodes
"""

from habitat.utils.visualizations.utils import images_to_video
import os
from imageio import imread
from tqdm import tqdm
import re

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)] 

def video_from_dir(vidDir, images_type="png"):
    """
    Images will be ordered alpha-numeric order for video
    vidDir: str directory path that has images

    Output video is called 'video.mp4' and placed in same directory
    """
    assert os.path.isdir(vidDir), "vidDir must be a valid directory"
    images = sorted(os.listdir(vidDir))
    images.sort(key=natural_sort_key)

    imArray = []
    for i in images:
        if not i.endswith('.' + images_type): continue
        tmp = imread(os.path.join(vidDir, i))
        # print(tmp)
        # print(type(tmp))
        # print(tmp.shape)
        imArray.append(tmp)
    images_to_video(imArray, vidDir, "video", fps=2)

def generate_videos(episodes_dir):
    """
    episodes_dir: str must contain directories with images
    """
    assert os.path.isdir(episodes_dir)
    dirs = os.listdir(episodes_dir)
    pbar = tqdm(dirs)
    for d in tqdm(dirs):
        pbar.set_description("processing dir {}/".format(d))
        full_d = os.path.join(episodes_dir, d)
        if not os.path.isdir(full_d): continue
        video_from_dir(full_d)
        # print(full_d)

generate_videos("outputs/dump/FinalTrainExp3/episodes/1/")



