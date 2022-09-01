import cv2
import os
from PIL import Image
from PIL.ExifTags import TAGS
# import flirimageextractor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg     
import flir_image_extractor
import subprocess
import json
import numpy as np

# TODO: convert entire to OOP
# TODO: notebook

# each item in images will be stored as tuple (image_file, filename)
# so that we remember filename for future use
# --> goal: to learn how to read images and store as 2D array of pixels
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append((img, filename))
    return images


# returns rows by cols in a tuple
def sizeof(array):
    return (len(array), len(array[0]))


# for now, converts any thermal image to digital and normalized thermal image
# https://github.com/Nervengift/read_thermal.py
def convert_img(image_path):
    flir = flir_image_extractor.FlirImageExtractor()
    flir.process_image(image_path)
    flir.save_images()
    flir.plot()
    return


# gets metadata from specified image_path
# returns metadata as a dict, making it easy to extract desired attributes
def extract_metadata(image_path):
    meta_json = subprocess.check_output(
            ['exiftool', '-j', image_path])
    meta = json.loads(meta_json.decode())[0]
    return meta


# takes in image path, and loads up an editor window
# where you can alter the image
# ultimate goal is to create faulty images and 
# use them to test our overall thermal fault detection system
def pixel_selector(image_path):
    def mouse_callback(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("value at position: ", x, y, "is: ", image[y][x])
            # for i in range(-5, 6):
            #     for j in range(-5, 6):
            #         image[y+i][x+j] = [255, 100, 100]
        return
    image = cv2.imread(os.path.join(image_path))
    rows, cols = sizeof(image)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL) # Can be resized
    cv2.resizeWindow('image', cols, rows) #Reasonable size window
    cv2.setMouseCallback('image', mouse_callback) #Mouse callback
    finished = False
    while(not finished):
        cv2.imshow('image', image)
        k = cv2.waitKey(4) & 0xFF
        if k == 27:
            finished = True
    cv2.destroyAllWindows()

    image_path = image_path.split('\\')
    image_path, filename = "\\".join(image_path[:-1]), image_path[-1]
    filename, extension = filename.split('.')

    # not ycrcb image
    if filename.count('_') == 1:
        prev_condition, scale = filename.split('_')
    # ycrcb image
    else:
        prev_condition, scale, colorspace = filename.split('_')

    # new_condition = input("enter new condition: ")
    new_condition = "test"
    if colorspace:
        new_filename = new_condition + "_" + scale + "_" + colorspace + '.' + extension
    else:
        new_filename = new_condition + '_' + scale + '.' + extension
    # cv2.imwrite(image_path + '\\' + new_filename, image)
    return


def bgr2ycrcb(image_path):
    filename = image_path.split('\\')[-1]
    filename, extension = filename.split('.')
    condition, scale = filename.split('_')[:2]

    print("For file: ", filename)
    print("Condition: ", condition)
    print("Scale: ", scale)
    print("=============================")
    img = cv2.imread(image_path)
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    new_filename = filename + "_ycrcb." + extension
    cv2.imwrite(os.path.join("YCRCB-Reference-Images", new_filename), ycrcb_img)
    return


# gets min pixel value, max pixel value, and minstep btwn pixels
# input should be grayscale images, algo will only look at first channel if 3 channel image
def find_stats(dir_path):
    # images = load_images_from_folder(path)
    def calc_stats(img):
        # converts 3 channels of grayscale to 1 channel, using the first channel
        # DO NOT USE WITH RGB IMAGES
        norm_img = []
        for i in range(len(img)):
            for j in range(len(img[0])):
                norm_img.append(int(img[i][j][0]))
        
        min_step = 255
        norm_img.sort()
        print(norm_img)
        min_val = norm_img[0]
        max_val = norm_img[-1]
        prev = norm_img[0]
        for i in range(1, len(norm_img)):
            curr_value = norm_img[i]
            curr_step = curr_value - prev
            if curr_step != 0:
                min_step = min(min_step, curr_step)
            prev = norm_img[i]
        return min_val, max_val, min_step
    for filename in os.listdir(dir_path):
        img = cv2.imread(os.path.join("thermal-images", filename[:-4] + "-thermal.png"))
        if img is None:
            flir = flir_image_extractor.FlirImageExtractor()
            flir.process_image(os.path.join(dir_path,filename))
            flir.get_thermal()
            img = cv2.imread(os.path.join("thermal-images", filename[:-4] + "-thermal.png"))
        min_val, max_val, min_step = calc_stats(img)
        print(filename)
        print("minimum: ", min_val, "maximum: ", max_val, "min_step: ", min_step)
        break

        # cv2.imshow('image', img)
        # cv2.namedWindow('image',cv2.WINDOW_NORMAL) # Can be resized
        # finished = False
        # while(not finished):
        #     cv2.imshow('image',img)
        #     k = cv2.waitKey(4) & 0xFF
        #     if k == 27:
        #         finished = True
        # cv2.destroyAllWindows()
    return


# for now, crops left and top of image to remove scale and FLIR label
def crop_image():
    images = []
    folder = "to-be-cropped"
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append((img, filename))
    for img in images:
        unpacked_filename = img[1].split('.')
        filename, extension = ".".join(unpacked_filename[:-1]), unpacked_filename[-1]
        img = img[0]
        rows, cols = sizeof(img)
        cropped_img = img[56:rows, 88:cols]
        cv2.imwrite(os.path.join("cropped-images", filename + "-cropped." + extension), cropped_img)
    return

def ycrcb2y(image_path):
    filename = image_path.split('\\')[-1]
    filename, extension = filename.split('.')
    condition, scale = filename.split('_')[:2]
    img = cv2.imread(image_path)
    Y_image = img[:,:,0]

    cv2.imwrite(os.path.join("Y-Reference-Images", condition + '_' + scale + "_Y." + extension), Y_image)
    return

if __name__ == "__main__":
    # for filename in os.listdir("Reference-Images"):
    #     bgr2ycrcb(os.path.join("Reference-images", filename))

    for filepath in os.listdir('YCRCB-Reference-Images'):
        ycrcb2y(os.path.join("YCRCB-Reference-Images", filepath))

    # pixel_selector(os.path.join("Reference-Images", "Heavy_27-52.jpg"))
    # pixel_selector(os.path.join("YCRCB-Reference-Images", "Heavy_27-52_ycrcb.jpg"))
    # pixel_selector(os.path.join("Y-Reference-Images", "Heavy_27-52_Y.jpg"))
    #convert_img(os.path.join("Reference-Images", "Heavy_27-52.jpg"))
    # print(extract_metadata(os.path.join("Reference-Images", "Heavy_27-52.jpg")))
    pass