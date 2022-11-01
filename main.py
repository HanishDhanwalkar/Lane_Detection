import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob

image_path =input("Enter the path of the test image(without \"__\") ")

test_images = [plt.imread(img) for img in glob.glob(image_path)] #test_images = [plt.imread(img) for img in glob.glob('test_images/*.jpg')]

def list_images(images, cols = 2, rows = 5, cmap=None):
    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        #Use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap = cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

def RGB_color_selection(image):

    #White color mask
    lower_threshold = np.uint8([200,200,200])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    
    #offwhite color mask
    lower_threshold = np.uint8([175, 175,   0])
    upper_threshold = np.uint8([255, 255, 255])
    offwhite_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    
    #Combine white and offwhite masks
    mask = cv2.bitwise_or(white_mask, offwhite_mask)
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    
    return masked_image

# list_images(list(map(RGB_color_selection, test_images)))
images_after_RGB = list(map(RGB_color_selection, test_images))

def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

gray_images = list(map(gray_scale, images_after_RGB))
# list_images(gray_images)

def gaussian_smoothing(image, kernel_size = 13):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

blur_images = list(map(gaussian_smoothing, gray_images))


def canny_detector(image, low_threshold = 50, high_threshold = 150):
    return cv2.Canny(image, low_threshold, high_threshold)

edge_detected_images = list(map(canny_detector, blur_images))
# list_images(edge_detected_images)

def region_selection_leftlane(image):

    mask = np.zeros_like(image)   
    #Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.001, rows * 0.99]
    top_left     = [cols * 0.001, rows * 0.6]
    bottom_right = [cols * 0.3, rows * 0.95]
    top_right    = [cols * 0.3, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def region_selection_rightlane(image):

    mask = np.zeros_like(image)   
    #Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.5, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.75]
    bottom_right = [cols * 0.99, rows * 0.99]
    top_right    = [cols * 0.6, rows * 0.75]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def region_selection_middlelane(image):

    mask = np.zeros_like(image)   
    #Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.3, rows * 0.99]
    top_left     = [cols * 0.3, rows * 0.8]
    bottom_right = [cols * 0.5, rows * 0.99]
    top_right    = [cols * 0.7, rows * 0.8]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
# list_images(masked_image)

def hough_transform(image):

    rho = 1              #Distance resolution of the accumulator in pixels.
    theta = np.pi/180    #Angle resolution of the accumulator in radians.
    threshold = 20       #Only lines that are greater than threshold will be returned.
    minLineLength = 20   #Line segments shorter than that are rejected.
    maxLineGap = 300     #Maximum allowed gap between points on the same line to link them
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold, minLineLength = minLineLength, maxLineGap = maxLineGap)


# print(hough_lines)


masked_image_left = list(map(region_selection_leftlane, edge_detected_images))
hough_lines_left = list(map(hough_transform, masked_image_left))
masked_image_right = list(map(region_selection_rightlane, edge_detected_images))
hough_lines_right = list(map(hough_transform, masked_image_right))
masked_image_middle = list(map(region_selection_middlelane, edge_detected_images))
hough_lines_middle = list(map(hough_transform, masked_image_middle))

def draw_lines_left(image, lines, thickness = 4):
    color = [255,0,0]
    image = np.copy(image)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def draw_lines_rightt(image, lines, thickness = 4):
    color = [0,255,0]
    image = np.copy(image)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def draw_lines_middle(image, lines, thickness = 4):
    color = [0,0,255]
    image = np.copy(image)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

line_images = []

for image in test_images:
    for lines in hough_lines_left:
        image=draw_lines_left(image,lines)
    for lines in hough_lines_right:
        image=draw_lines_rightt(image, lines)
    for lines in hough_lines_middle:
        image=draw_lines_middle(image,lines)
    
    line_images.append(image)


list_images(line_images)