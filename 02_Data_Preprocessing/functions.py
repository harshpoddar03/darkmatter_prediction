import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob


# Load the image from a file
def prepro(x):
# Get the dimensions of the image
    height, width = x.shape[:2]

    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Define the crop size and calculate the start point for cropping
    crop_size = 85
    start_x = center_x - crop_size // 2
    start_y = center_y - crop_size // 2

    # Crop the image
    cropped_image = x[start_y:start_y+crop_size, start_x:start_x+crop_size]
    return cropped_image


#  Create a function that takes in a list of images and displays them in a grid
def show_images(images, title):
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    for i, image in enumerate(images):
        axs[i].imshow(image)
        axs[i].axis('off')
    plt.show()



import numpy as np
from scipy.ndimage import center_of_mass


def centroid(image):
    # Convert to grayscale if the image is in color
    if image.ndim == 3 and image.shape[2] in [3, 4]:  # Checking for RGB or RGBA
        image = image.mean(axis=2)  # Converting to grayscale by averaging channels
    
    # Calculate the center of mass, which is the intensity-weighted centroid of the image
    centroid = center_of_mass(image)
    return centroid


## optimised the data_processing code 400 times faster by vectorizing the code

def calculate_radiuses():
    radiuses = {}
    for rad in range(200):

        x_bases_array =[]
        y_bases_array = []

        for degree in range(360):
                radian = np.deg2rad(degree)
                x_base = rad * np.cos(radian)
                y_base = rad * np.sin(radian)
                x_bases_array.append(x_base)
                y_bases_array.append(y_base)


        x_bases_array = np.array(x_bases_array).astype(int)
        y_bases_array = np.array(y_bases_array).astype(int)

        radiuses[rad] = (x_bases_array, y_bases_array)

    return radiuses




radiuses = calculate_radiuses()

def draw_circle_and_calculate_intensity(img, center, radius):

    
    

    x_peremeters_360 = radiuses[radius][0] + center[0]
    y_peremeters_360 = radiuses[radius][1] + center[1]

    # x_peremeters_360 = (radiuses[radius][0] + center[0]).astype(int)
    # y_peremeters_360 = (radiuses[radius][1] + center[1]).astype(int)


    x_peremeters_360_start = x_peremeters_360 - 2
    y_peremeters_360_start = y_peremeters_360 - 2

    x_peremeters_360_end = x_peremeters_360 + 3
    y_peremeters_360_end = y_peremeters_360 + 3

    x_peremeters_360_start = np.maximum(x_peremeters_360_start, 0)
    y_peremeters_360_start = np.maximum(y_peremeters_360_start, 0)
    x_peremeters_360_end = np.minimum(x_peremeters_360_end, img.shape[1])
    y_peremeters_360_end = np.minimum(y_peremeters_360_end, img.shape[0])

    # intensity_values = [ np.mean(img[y_peremeters_360_start[i]:y_peremeters_360_end[i], x_peremeters_360_start[i]:x_peremeters_360_end[i]]) for i in range(360)]
    
    ## best till now
    intensity_values = [ img[y_peremeters_360_start[i]:y_peremeters_360_end[i], x_peremeters_360_start[i]:x_peremeters_360_end[i]].mean() for i in range(360)]


    return intensity_values

from tqdm import tqdm

def get_images(directory):
    # Get all the npy files in the directory
    files = glob.glob(directory + '/*.npy')
    
    data = []
    for filename in tqdm(files, desc='Loading images', unit='image', total=len(files)):
        with open(filename, 'rb') as f:
            # Load the npy file content into a numpy array
            data.append(np.load(f))
    return data
    
        