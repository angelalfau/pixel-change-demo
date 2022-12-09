from lzma import FILTER_LZMA1
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

from scipy import ndimage
from PIL import Image

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
        radius = 2
        if event == cv2.EVENT_LBUTTONDOWN:
            # print("value at position: ", x, y, "is: ", image[y][x])
            print("altered center: ", y, x)
            print("# of altered pixels: ", (radius*2+1)**2)
            for i in range(radius*-1, radius+1):
                for j in range(radius*-1, radius+1):
                    image[y+i][x+j] = [255, 255, 255]
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

    return image

    filename = image_path.split('\\')[-1]
    filename, extension = filename.split('.')

    # not ycrcb image
    if filename.count('_') == 1:
        prev_condition, scale = filename.split('_')
    # ycrcb image
    else:
        prev_condition, scale, colorspace = filename.split('_')

    if colorspace:
        new_filename = prev_condition + "_" + scale + "_" + colorspace + "_altered." + extension
    else:
        new_filename = prev_condition + '_' + scale + "_altered." + extension
    
    shouldSave = input("save image? (y/n): ")
    if shouldSave == 'y':
        cv2.imwrite(os.path.join("altered-images", new_filename), image)
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
def crop_image(img):
    rows, cols = sizeof(img)
    cropped_img = img[:, :521]
    return cropped_img

def ycrcb2y(image_path):
    filename = image_path.split('\\')[-1]
    filename, extension = filename.split('.')
    img = cv2.imread(image_path)
    Y_image = img[:,:,0]

    cv2.imwrite(os.path.join("Y-Reference-Images", filename[:-6] + "_Y." + extension), Y_image)
    return

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

def registration(img1, img2, img1_color, mask=None):
    height, width = sizeof(img1)
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)
    
    # Trying to do edge detection and registration using edge detection
    # then using the resulting matrix on the original image
    threshold1 = 30
    threshold2 = 45
    apertureSize = 3

    mask1 = cv2.Canny(img1, threshold1, threshold2, apertureSize)
    mask2 = cv2.Canny(img2, threshold1, threshold2, apertureSize)

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

def show_registration(rotation_folder, reference_folder, rotation_filename, reference_filename):
    img1_color = crop_image(cv2.imread(os.path.join(rotation_folder, rotation_filename)))
    img2_color = crop_image(cv2.imread(os.path.join(reference_folder, reference_filename)))
    img1_y = cv2.cvtColor(img1_color, cv2.COLOR_BGR2YCrCb)[:,:,0]
    img2_y = cv2.cvtColor(img2_color, cv2.COLOR_BGR2YCrCb)[:,:,0]
    registered_img = registration(img1_y, img2_y, img1_color)
    # cv2.imwrite(os.path.join("post-registration", "Heavy+Normal_27-52_Y_equalized_heavy_nowidth.jpg"), registered_img)
    finished = False
    while(not finished):
        cv2.imshow('registered rotations', np.hstack([img1_color, img2_color, registered_img]))
        k = cv2.waitKey(4) & 0xFF
        if k == 27:
            finished = True
    return registered_img

def img_subtraction(img1, img2, shouldShow = True):
    img3 = cv2.absdiff(img1,img2)
    if shouldShow:
        finished = False
        while(not finished):
            cv2.imshow('image subtraction', np.hstack([img1, img2, img3]))
            k = cv2.waitKey(4) & 0xFF
            if k == 27:
                finished = True
    return img3

def detect_faults(img, detection_criteria = 100):
    rows, cols = sizeof(img)
    tmp = img.copy()

    isSingleChannel = type(img[0,0]) == np.uint8
    
    for i in range(rows):
        for j in range(cols):
            if (isSingleChannel and tmp[i,j] >= detection_criteria) or (not isSingleChannel and tmp[i,j,0] >= detection_criteria):
                print(i, j)
                queue = [(i,j)]
                seen = set([(i,j)])
                while queue:
                    x,y = queue.pop(0)
                    if (isSingleChannel and tmp[x,y] >= detection_criteria) or (not isSingleChannel and tmp[x,y,0] >= detection_criteria):
                        
                        if isSingleChannel:
                            tmp[x,y] = 0
                        else:
                            tmp[x,y] = [0,0,0]
                        seen.add((x,y))
                        if x < rows-1:
                            queue.append((x+1,y))
                        if x > 0:
                            queue.append((x-1,y))
                        if y < cols-1:
                            queue.append((x,y+1))
                        if y > 0:
                            queue.append((x,y-1))
                center_x = 0
                center_y = 0
                min_x = rows
                max_x = 0
                min_y = cols
                max_y = 0
                for i,j in seen:
                    center_x += i
                    center_y += j
                    min_x = min(min_x, i)
                    max_x = max(max_x, i)
                    min_y = min(min_y, j)
                    max_y = max(max_y, j)
                center_x //= len(seen)
                center_y //= len(seen)
                curr_radius = max(max_x-min_x, max_y-min_y) // 2 + 20
                # currently making the circle white. May want to revert back to red circle
                # However, changing back to red would mean having to convert single-channel images to 3-channel images
                # and then isSingleChannel can also be removed
                print("detected center: ", center_x, center_y)
                print("detected pixels changed: ", len(seen))
                cv2.circle(img, center = (center_y, center_x), radius=curr_radius, color=(255,255,255), thickness=2)

    def mouse_callback(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("value at position: ", x, y, "is: ", img[y][x])
        return

    cv2.namedWindow('image', cv2.WINDOW_NORMAL) # Can be resized
    cv2.resizeWindow('image', cols, rows) #Reasonable size window
    cv2.setMouseCallback('image', mouse_callback) #Mouse callback
    finished = False
    while(not finished):
        cv2.imshow('image', img)
        k = cv2.waitKey(4) & 0xFF
        if k == 27:
            finished = True
    cv2.destroyAllWindows()

    return img

def show_registration_and_subtraction(altered_img, orig_img):
    registered_img = registration(altered_img, orig_img, altered_img)
    subtracted_img = img_subtraction(registered_img, orig_img, False)

    finished = False
    while(not finished):
        cv2.imshow('image workflow', np.hstack([altered_img, registered_img, subtracted_img]))
        k = cv2.waitKey(4) & 0xFF
        if k == 27:
            finished = True
    return subtracted_img

def add_fault(img, fault_center, lum_diff, radius):
    y, x = fault_center
    for i in range(radius*-1, radius+1):
        for j in range(radius*-1, radius+1):
            for k in range(len(img[0,0])):
                img[y+i][x+j][k] = min(255, img[y+i][x+j][k] + lum_diff)
    return img

def rotate_image(path, angle):
    to_rotate = Image.open(path)
    rotated = to_rotate.rotate(angle)
    rotated.save(os.path.join("temp", "WH_Normal_26-52_rotated_altered.jpg"))
    return

def detect_fault_error(subtracted, lum_diff, fault_radius, fault_center, num_pixels_error, lum_diff_error, size_err_by_deg, lum_err_by_deg, rotation_indx, lum_diff_indx, radius_indx):
    rows, cols = sizeof(subtracted)
    tmp = subtracted.copy()

    radius_to_check = fault_radius + 10
    detection_criteria = 50
    isSingleChannel = type(subtracted[0,0]) == np.uint8
    
    for i in range(fault_center[0]-radius_to_check, fault_center[0]+radius_to_check):
        for j in range(fault_center[1]-radius_to_check, fault_center[1]+radius_to_check):
            if (isSingleChannel and tmp[i,j] >= detection_criteria) or (not isSingleChannel and tmp[i,j,0] >= detection_criteria):
                detected_lum_diff = 0
                queue = [(i,j)]
                seen = set([(i,j)])
                while queue:
                    x,y = queue.pop(0)
                    if (isSingleChannel and tmp[x,y] >= detection_criteria) or (not isSingleChannel and tmp[x,y,0] >= detection_criteria):
                        detected_lum_diff += tmp[x,y]
                        if isSingleChannel:
                            tmp[x,y] = 0
                        else:
                            tmp[x,y] = [0,0,0]
                        seen.add((x,y))
                        if x < rows-1:
                            queue.append((x+1,y))
                        if x > 0:
                            queue.append((x-1,y))
                        if y < cols-1:
                            queue.append((x,y+1))
                        if y > 0:
                            queue.append((x,y-1))
                center_x = 0
                center_y = 0
                min_x = rows
                max_x = 0
                min_y = cols
                max_y = 0
                for i,j in seen:
                    center_x += i
                    center_y += j
                    min_x = min(min_x, i)
                    max_x = max(max_x, i)
                    min_y = min(min_y, j)
                    max_y = max(max_y, j)
                center_x //= len(seen)
                center_y //= len(seen)

                center_error = (abs(center_x-fault_center[0]), abs(center_y-fault_center[1]))

                actual_size = (fault_radius*2+1)**2
                size_error = abs(len(seen) - actual_size) / actual_size

                # print(len(seen), actual_size, size_error)
                # if size_error > 100:
                #     print("========\nsize error too large")
                #     print("rotation: ", rotation_deg)
                #     print("lum diff: ", lum_diff)
                #     print("radius: ", fault_radius)

                detected_lum_diff /= len(seen)
                lum_error = abs(detected_lum_diff - lum_diff) / lum_diff

                # (subtracted, lum_diff, fault_radius, fault_center, num_pixels_error, lum_diff_error, size_err_by_deg, lum_err_by_deg, rotation_indx, lum_diff_indx, radius_indx)
                num_pixels_error[radius_indx] += size_error
                lum_diff_error[lum_diff_indx] += lum_error
                size_err_by_deg[rotation_indx] += size_error
                lum_err_by_deg[rotation_indx] += lum_error
                center_x_err_by_deg[rotation_indx] += center_error[0]
                center_y_err_by_deg[rotation_indx] += center_error[1]

                aspect_ratio = (max_x-min_x+1) / (max_y-min_y+1)

                print(aspect_ratio)

                if aspect_ratio < 0.5 or aspect_ratio > 2:
                    print("========\naspect ratio too large")
                    print("rotation: ", rotation_deg)
                    print("lum diff: ", lum_diff)
                    print("radius: ", fault_radius)
                    print("size error: ", size_error)
                    print("lum error: ", lum_error)
                    print("center error: ", center_error)
                    print("aspect ratio: ", aspect_ratio)
                    show_img(subtracted)
                # if size_error > 40:
                #     print(len(seen), size_error)
                #     show_img(subtracted)
                # print("size error: ", size_error)
                return aspect_ratio
    # print("no fault detected", rotation_deg, lum_diff, fault_radius)
    # show_img(tmp)
    return 0

def show_img(img, title='title'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ### Saves cropped images and converts to different color spaces
    # for filename in os.listdir("reference-images"):
    #     cropped_img = crop_image(cv2.imread(os.path.join("reference-images", filename)))
    #     cv2.imwrite(os.path.join("cropped-reference-images", filename) , cropped_img)
    #     bgr2ycrcb(os.path.join("reference-images", filename))
    #     ycrcb2y(os.path.join("ycrcb-reference-images", filename[:-4] + "_ycrcb.jpg"))


    ### IMAGE REGISTRATION
    # show_registration(rotation_folder = "rotation-images", reference_folder = "reference-images", rotation_filename = "Iron_Heavy_26-52_r4.jpg", reference_filename = "Iron_Heavy_26-52.jpg")
    

    ### CREATING PRE-PERTURBATION IMAGES
    # img = cv2.imread(os.path.join("reference-images", "WH_Normal_26-52.jpg"))
    # cv2.imwrite(os.path.join("pre-perturbation", "WH_Normal_26-52.jpg"), img)
    
    # to_rotate = Image.open("./pre-perturbation/WH_Normal_26-52.jpg")
    # rotated = to_rotate.rotate(15)
    # rotated.save("./pre-perturbation/WH_Normal_26-52_rotated.jpg")

    # img = cv2.imread(os.path.join("pre-perturbation", "WH_Normal_26-52.jpg"))
    # offset = np.zeros_like(img)
    # offset_amt = 140
    # isOffsetRight = True
    # if isOffsetRight:
    #     offset[:,offset_amt:] = img[:,:-offset_amt]    
    # else:
    #     offset[:,:-offset_amt] = img[:,offset_amt:]
    # cv2.imwrite(os.path.join("pre-perturbation", "WH_Normal_26-52_offset.jpg"), offset)
    # altered_img = pixel_selector(os.path.join("pre-perturbation", "WH_Normal_26-52_offset.jpg"))
    # cv2.imwrite(os.path.join("post-perturbation", "WH_Normal_26-52_offset_altered.jpg"), altered_img)

    # rotated = cv2.imread(os.path.join("pre-perturbation", "WH_Normal_26-52_rotated.jpg"))
    # offset_rotated = np.zeros_like(rotated)
    # offset_rotated[:,100:] = rotated[:,:-100]
    # cv2.imwrite(os.path.join("pre-perturbation", "WH_Normal_26-52_rotated_offset.jpg"), offset_rotated)


    ### CREATING POST-PERTURBATION IMAGES
    # pre_perturbation_folder = "pre-perturbation"
    # post_perturbation_folder = "post-perturbation"
    # for filename in os.listdir(pre_perturbation_folder):
    #     altered_img = pixel_selector(os.path.join(pre_perturbation_folder, filename))
    #     cv2.imwrite(os.path.join(post_perturbation_folder, filename[:-4] + '_altered.jpg'), altered_img)
    
    
    ### REGISTRATION OF POST-PERTURBATION IMAGES
    # for filename in os.listdir("post-perturbation"):
    #     show_registration(rotation_folder = "post-perturbation", reference_folder = "reference-images", rotation_filename = filename, reference_filename = "WH_Normal_26-52.jpg")
    

    ### RE-DOING rotated image
    # to_rotate = Image.open("./pre-perturbation/WH_Normal_26-52.jpg")
    # rotated = to_rotate.rotate(15)
    # rotated.save("./pre-perturbation/WH_Normal_26-52_rotated.jpg")
    # rotated = cv2.imread(os.path.join("pre-perturbation", "WH_Normal_26-52_rotated.jpg"))
    # altered_img = pixel_selector(os.path.join("pre-perturbation", "WH_Normal_26-52_rotated.jpg"))
    # cv2.imwrite(os.path.join("post-perturbation", "WH_Normal_26-52_rotated_altered.jpg"), altered_img)


    ### RE-DOING offset and rotated image
    # rotated = cv2.imread(os.path.join("pre-perturbation", "WH_Normal_26-52_rotated.jpg"))
    # offset_rotated = np.zeros_like(rotated)
    # offset_rotated[:,50:] = rotated[:,:-50] # offset to the right
    # # offset_rotated[:,:-50] = rotated[:,50:] # offset to the left
    # cv2.imwrite(os.path.join("pre-perturbation", "WH_Normal_26-52_rotated_offset.jpg"), offset_rotated)
    # altered_img = pixel_selector(os.path.join("pre-perturbation", "WH_Normal_26-52_rotated_offset.jpg"))
    # cv2.imwrite(os.path.join("post-perturbation", "WH_Normal_26-52_rotated_offset_altered.jpg"), altered_img)


    # detect_faults(cv2.imread(os.path.join("post-perturbation", "WH_Normal_26-52_altered.jpg")))


    # orig_img = cv2.imread(os.path.join("pre-perturbation", "WH_Normal_26-52.jpg"))
    # orig_perturbed_img = cv2.imread(os.path.join("post-perturbation", "WH_Normal_26-52_altered.jpg"))
    # rotated_perturbed_img = cv2.imread(os.path.join("post-perturbation", "WH_Normal_26-52_rotated_altered.jpg"))
    # show_registration(rotation_folder = "post-perturbation", reference_folder = "edge-detected", rotation_filename = "WH_Normal_26-52_rotated_altered.jpg", reference_filename = "WH_Normal_26-52_edges.jpg")
    # registration(rotated_perturbed_img, orig_img, rotated_perturbed_img)
    # subtracted_img = show_registration_and_subtraction(rotated_perturbed_img, orig_img)
    # detect_faults(subtracted_img)


    ### Loading in images
    # orig_img = cv2.imread(os.path.join("pre-perturbation", "WH_Normal_26-52.jpg"))
    # rotated_img = cv2.imread(os.path.join("post-perturbation", "WH_Normal_26-52_rotated_altered.jpg"))
    
    # print(sizeof(orig_img))

    # ### Doing canny edge detection
    # t_lower = 30
    # t_upper = 45
    # aperture_size = 1
    # orig_edge = cv2.Canny(orig_img, t_lower, t_upper, apertureSize=aperture_size)
    # show_img(orig_edge, "orig_edge")
    # rotated_edge = cv2.Canny(rotated_img, t_lower, t_upper, apertureSize=aperture_size)

    # ### Registration of Edge Detected Images
    # registered_edge = registration(rotated_edge, orig_edge, rotated_img[:, :, 0])
    # registered_img = registration(rotated_img, orig_img, rotated_img)

    # ### Doing Fault Detection and Displaying Result
    # singleChannel_orig_img = orig_img[:, :, 0]
    # subtracted_img = cv2.absdiff(registered_edge, singleChannel_orig_img)
    # detect_faults(subtracted_img)

    # ### Displays Registration
    # cv2.imshow('original', np.hstack([orig_img, rotated_img, registered_img]))
    # cv2.imshow('edge', np.hstack([orig_edge, rotated_edge, registered_edge]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # altered_img = pixel_selector(os.path.join("pre-perturbation", "WH_Normal_26-52.jpg"))
    # cv2.imwrite(os.path.join("post-perturbation", "WH_Normal_26-52_altered.jpg"), altered_img)

    # to_rotate = Image.open("./post-perturbation/WH_Normal_26-52_altered.jpg")
    # rotated = to_rotate.rotate(15)
    # rotated.save("./post-perturbation/WH_Normal_26-52_rotated_altered.jpg")


    # Rotation Parameters (0, 45, 5)
    min_deg = 0
    max_deg = 45
    deg_step = 5

    # Perturbation Value Parameters (50, 100, 10)
    min_lum_diff = 50
    max_lum_diff = 100
    lum_diff_step = 10

    # Perturbation Size Parameters (0, 3, 1)
    min_fault_radius = 0
    max_fault_radius = 3
    fault_radius_step = 1
    
    # Edge Detection Parameters (30, 45, 3)
    t_lower = 30
    t_upper = 45
    aperture_size = 3

    average_aspect_ratio = 0.0

    num_pixels_x = [(radius*2+1)**2 for radius in range(min_fault_radius, max_fault_radius + 1, fault_radius_step)]
    num_pixels_error = [0 for _ in range(len(num_pixels_x))]

    lum_diff_x = [lum_diff for lum_diff in range(min_lum_diff, max_lum_diff+1, lum_diff_step)]
    lum_diff_error = [0 for _ in range(len(lum_diff_x))]

    rotation_x = [deg for deg in range(min_deg, max_deg+1, deg_step)]
    size_err_by_deg = [0 for _ in range(len(rotation_x))]
    lum_err_by_deg = [0 for _ in range(len(rotation_x))]
    center_x_err_by_deg = [0 for _ in range(len(rotation_x))]
    center_y_err_by_deg = [0 for _ in range(len(rotation_x))]

    fault_center = (340, 75)

    for lum_diff in range(min_lum_diff, max_lum_diff+1, lum_diff_step):
        lum_diff_indx = (lum_diff // lum_diff_step) - (min_lum_diff // lum_diff_step)

        for fault_radius in range(min_fault_radius, max_fault_radius+1, fault_radius_step):
            radius_indx = fault_radius // fault_radius_step

            orig_img = cv2.imread(os.path.join("pre-perturbation", "WH_Normal_26-52.jpg"))
            single_channel_orig = orig_img[:, :, 0]
            orig_edge = cv2.Canny(orig_img, t_lower, t_upper, apertureSize=aperture_size)

            img = add_fault(orig_img, fault_center, lum_diff, fault_radius)
            cv2.imwrite(os.path.join("temp", "WH_Normal_26-52_altered.jpg"), img)

            for rotation_deg in range(min_deg, max_deg+1, deg_step):
                rotation_indx = rotation_deg // deg_step

                rotate_image(os.path.join("temp", "WH_Normal_26-52_altered.jpg"), rotation_deg)
                rotated = cv2.imread(os.path.join("temp", "WH_Normal_26-52_rotated_altered.jpg"))
                rotated_edge = cv2.Canny(rotated, t_lower, t_upper, apertureSize=aperture_size)
                registered = registration(rotated_edge, orig_edge, rotated[:, :, 0])
                subtracted = cv2.subtract(registered, single_channel_orig)

                average_aspect_ratio += detect_fault_error(subtracted, lum_diff, fault_radius, fault_center, num_pixels_error, lum_diff_error, size_err_by_deg, lum_err_by_deg, rotation_indx, lum_diff_indx, radius_indx)
                # print("radius: ", fault_radius, "\nlum diff: ", lum_diff)
                # detect_faults(subtracted, 50)

    average_aspect_ratio /= 240
    print("average aspect ratio: ", average_aspect_ratio)

    for i in range(len(num_pixels_error)):
        num_pixels_error[i] /= 60
        num_pixels_error[i] *= 100
    for i in range(len(lum_diff_error)):
        lum_diff_error[i] /= 40
        lum_diff_error[i] *= 100
    for i in range(len(size_err_by_deg)):
        size_err_by_deg[i] /= 24
        lum_err_by_deg[i] /= 24
        center_x_err_by_deg[i] /= 24
        center_y_err_by_deg[i] /= 24

        size_err_by_deg[i] *= 100
        lum_err_by_deg[i] *= 100

    plt.plot(num_pixels_x, num_pixels_error)
    plt.xlabel("Fault Size (Pixels)")
    plt.ylabel("Fault Size Error %")
    plt.title("Fault Size Error Percentage vs Fault Size")
    plt.show()

    plt.plot(lum_diff_x, lum_diff_error)
    plt.xlabel("Luminance Difference")
    plt.ylabel("Luminance Difference Error %")
    plt.title("Luminance Difference Error Percentage vs Luminance Difference")
    plt.show()

    plt.plot(rotation_x, size_err_by_deg)
    plt.xlabel("Amount of Rotation (Degrees)")
    plt.ylabel("Avg Fault Size Error %")
    plt.title("Fault Size Error vs Amount of Rotation")
    plt.show()

    plt.plot(rotation_x, lum_err_by_deg)
    plt.xlabel("Amount of Rotation (Degrees)")
    plt.ylabel("Avg Luminance Difference Error %")
    plt.title("Luminance Difference Error vs Amount of Rotation")
    plt.show()

    plt.plot(rotation_x, center_x_err_by_deg)
    plt.xlabel("Amount of Rotation (Degrees)")
    plt.ylabel("Horizontal Center Error (Pixels)")
    plt.title("Horizontal Center Error vs Amount of Rotation")
    plt.show()

    plt.plot(rotation_x, center_y_err_by_deg)
    plt.xlabel("Amount of Rotation (Degrees)")
    plt.ylabel("Vertical Center Error (Pixels)")
    plt.title("Vertical Center Error vs Amount of Rotation")
    plt.show()
    pass


