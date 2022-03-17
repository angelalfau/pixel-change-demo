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

# pass in array of image data files (each file is array of pixels)
# only used for jpg files!!!
# creates white square in middle of image
# and saves to altered-images folder
# --> goal: to learn how to alter images and save data files back to images
# after altering images, metadata is lost ==> maybe find way to read or save metadata
def alter_images(orig_images):
    for orig_image in orig_images:
        filename, extension = orig_image[1].split('.')
        image = orig_image[0]
        print("currently reading: ", filename, ".", extension, sep="")
        rows = len(image)
        cols = len(image[0])
        print("image is size: ", cols, "x", rows)

        if extension != 'jpg':
            print("unexpected file extension: ", extension)

        N = 1 # was prev rows//2
        M = 1 # was prev cols//2
        center_i = rows // 2
        center_j = cols // 2
        for i in range(-N, N):
            for j in range(-M, M):
                print("altering pixel at: ", center_i + i, center_j + j)
                image[center_i + i][center_j + j] = [255, 255, 255]
        
        new_filename = filename + "-altered." + extension
        cv2.imwrite(os.path.join("altered-images", new_filename), image)

# returns rows by cols in a tuple
def sizeof(array):
    return (len(array), len(array[0]))

# given two images, will find and print pixels that are diff by certain amt
def check_differences(image1, image2):
    if sizeof(image1) != sizeof(image2):
        print("images are of different size")
        return []
    diff_pixels = []
    for i in range(len(image1)):
        for j in range(len(image1[0])):
            # print(image1[i][j], image2[i][j])
            difference = abs(sum(image1[i][j]) - sum(image2[i][j]))
            if difference > 200:
                diff_pixels.append((i, j))
    return diff_pixels


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


def pixel_selector(image_path):
    # mouse callback function
    def mouse_callback(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            # print(image[0:100][0:100])
            # image[y][x] = [255, 255, 255]
            for i in range(-5, 6):
                for j in range(-5, 6):
                    image[y+i][x+j] = [255, 255, 255]
            # print(image[y][x])
        return
    image = cv2.imread(os.path.join(image_path))
    # image = mpimg.imread(os.path.join(image_path))
    # plt.imshow(image)
    # plt.show()
    rows, cols = sizeof(image)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL) # Can be resized
    cv2.resizeWindow('image', cols, rows) #Reasonable size window
    cv2.setMouseCallback('image', mouse_callback) #Mouse callback
    finished = False
    while(not finished):
        cv2.imshow('image',image)
        k = cv2.waitKey(4) & 0xFF
        if k == 27:
            finished = True
    print(image[0][0])
    cv2.destroyAllWindows()

'''
orig_images = load_images_from_folder("orig-images")
alter_images(orig_images)

iron_image = cv2.imread(os.path.join("orig-images", "Iron.jpg"))
iron_altered_image = cv2.imread(os.path.join("altered-images", "Iron-altered.jpg"))
# print(iron_image)
print(check_differences(iron_image, iron_altered_image))
'''

# rb_metadata = extract_metadata(os.path.join("orig-images", "Rainbow.jpg"))
# print(rb_metadata)

# print(convert_img(os.path.join("orig-images", "Rainbow.jpg")))

pixel_selector(os.path.join("orig-images", "Rainbow.jpg"))