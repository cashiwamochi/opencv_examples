import cv2
import numpy as np
import sys

from panorama import *

if __name__=="__main__":
    image_names = sys.argv
    if len(image_names) == 1:
        print("usage: python3 panorama_main.py image1 image2 image3 ...")
        exit()

    if len(image_names) == 2:
        print("usage: Plese give 2 or more images to this program.")
        exit()

    images = []
    panorama = []
    for i in range(1,len(image_names)):
        print( "Loading " + str(image_names[i]))
        img = resize_image(cv2.imread(image_names[i],cv2.IMREAD_COLOR))
        images.append(Image(str(i), img))

    panorama.append(Image(images[0].name, images[0].image))

    print("Your images have been loaded. Generating panorama starts ...")
    for i in range(0,len(images)-1):
        panorama.append(Image(str(i+1),make_panorama(panorama[i],images[i+1])))

    print("A panorama image is generated.")
    cv2.imwrite("panorama.png",panorama[-1].image)
