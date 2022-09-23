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
    return ycrcb_img


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
    return Y_image

def histogram_equalize(image_path):
    img = cv2.imread(image_path)

    filename = image_path.split('\\')[-1]
    filename, extension = filename.split('.')
    new_filename = filename + "_equalized." + extension

    hist,bins = np.histogram(img.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    equalized_img = cdf[img]

    cv2.imwrite(os.path.join("post-equalize", new_filename), equalized_img)

    print("saved equalized image to: ", os.path.join("post-equalize", new_filename))

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.title(filename)
    plt.show()

def registration(img1, img2, img1_color):
    height, width = sizeof(img1)
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)
    
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
    
    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    # Match the two sets of descriptors.
    matches = list(matcher.match(d1, d2))

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)
    
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)
    
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt
    
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    
    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                        homography, (width, height))

    # Save the output.
    return transformed_img
    # cv2.imwrite(os.path.join('output.jpg'), transformed_img)

if __name__ == "__main__":
    #for filename in os.listdir("Reference-Images"):
         #bgr2ycrcb(os.path.join("Reference-images", filename))
    #print(os.listdir("Reference-Images"))
    # for filepath in os.listdir('YCRCB-Reference-Images'):
    #     ycrcb2y(os.path.join("YCRCB-Reference-Images", filepath))

    #for filename in os.listdir("Training-Images"):
         #bgr2ycrcb(os.path.join("Training-images", filename))
    #for filepath in os.listdir('YCRCB-Reference-Images'):
         #ycrcb2y(os.path.join("YCRCB-Reference-Images", filepath))

    # pixel_selector(os.path.join("Reference-Images", "Heavy_27-52.jpg"))
    # pixel_selector(os.path.join("YCRCB-Reference-Images", "Heavy_27-52_ycrcb.jpg"))
    # pixel_selector(os.path.join("Y-Reference-Images", "Heavy_27-52_Y.jpg"))
    #convert_img(os.path.join("Reference-Images", "Heavy_27-52.jpg"))
    # print(extract_metadata(os.path.join("Reference-Images", "Heavy_27-52.jpg")))

    # histogram_equalize(os.path.join("Y-Reference-Images", "Heavy_27-52_Y.jpg"))
    # histogram_equalize(os.path.join("Y-Reference-Images", "Normal_27-52_Y.jpg"))

    img1 = cv2.imread(os.path.join("Training-Images", "rotation1.jpg"))
    y_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:,:,0]
    img1_color = cv2.imread(os.path.join("Training-Images", "rotation1.jpg"))
    img2 = cv2.imread(os.path.join("Training-Images", "rotation2.jpg"))
    y_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:,:,0]
    #img1 = ycrcb2y(bgr2ycrcb(img1))
    #img2 = ycrcb2y(bgr2ycrcb(img2))
    registered_img = registration(img1, img2, img1_color)
    cv2.imwrite(os.path.join("post-registration", "rotation_final.jpg"), registered_img)
    # img1 = cv2.imread(os.path.join("Y-Reference-Images", "Heavy_27-52_Y.jpg"))
    # img2 = cv2.imread(os.path.join("Y-Reference-Images", "Normal_27-52_Y.jpg"))
    # registered_img = registration(img1, img2, img1_color)
    # cv2.imwrite(os.path.join("post-registration", "Heavy+Normal_27-52_Y_heavy.jpg"), registered_img)

    # img_blur = cv2.imread(os.path.join("post-registration", "Heavy+Normal_27-52_Y_heavy.jpg"))
    # img2_blur = cv2.imread(os.path.join("post-registration", "Heavy+Normal_27-52_Y.jpg"))
    # # Canny Edge Detection
    # edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    # edges2 = cv2.Canny(image=img2_blur, threshold1=100, threshold2=200) # Canny Edge Detection

    # cv2.imwrite(os.path.join("post-canny", "Heavy+Normal_27-52_Y_heavy_canny.jpg"), edges)
    # cv2.imwrite(os.path.join("post-canny", "Heavy+Normal_27-52_Y_canny.jpg"), edges2)

    # Display Canny Edge Detection Image
    # cv2.imshow('Canny Edge Detection', edges)
    # cv2.imshow('Canny Edge Detection', edges2)
    # cv2.waitKey(0)


    pass
