import cv2
import os
from PIL import Image
from PIL.ExifTags import TAGS
import flirimageextractor
from matplotlib import cm

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

def sizeof(array):
    return (len(array), len(array[0]))

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

def extract_metadata(image_path):
    flir = flirimageextractor.FlirImageExtractor(palettes=[cm.jet, cm.bwr, cm.gist_ncar])
    # flir.process_image(image_path)
    flir.process_image('orig-images/Iron.jpg')
    flir.save_images()
    flir.plot()

    # image = Image.open(image_path)
    # exifdata = image.getexif()
    # print(exifdata)
    # for tag_id in exifdata:
    # # get the tag name, instead of human unreadable tag id
    #     tag = TAGS.get(tag_id, tag_id)
    #     data = exifdata.get(tag_id)
    #     # decode bytes 
    #     if isinstance(data, bytes):
    #         data = data.decode()
    #     print(f"{tag:25}: {data}")
    return

'''
orig_images = load_images_from_folder("orig-images")
alter_images(orig_images)

iron_image = cv2.imread(os.path.join("orig-images", "Iron.jpg"))
iron_altered_image = cv2.imread(os.path.join("altered-images", "Iron-altered.jpg"))
# print(iron_image)
print(check_differences(iron_image, iron_altered_image))
'''

# rb_image = cv2.imread(os.path.join("orig-images", "Rainbow.jpg"))
# exifdata = rb_image.getexif()

extract_metadata(os.path.join("orig-images", "Rainbow.jpg"))