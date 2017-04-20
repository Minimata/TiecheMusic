import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


def img_to_binary_grey_scale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # convert to grayscale
    ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    return thresh


def img_binary_vertical_hist(img):
    height, width = img.shape
    hist = [0] * height

    s = 0
    while s < width * height:
        hist[math.floor(s / width)] += 1 - img.item(s) / 255
        s += 1

    return hist


def img_binary_horizontal_hist(img):
    height, width = img.shape
    hist = [0] * width

    s = 0
    while s < width * height:
        hist[s % width] += 1 - img.item(s) / 255
        s += 1

    return hist


def display_hist(hist, title="histogram"):
    plt.xlabel('raw / column')
    plt.ylabel('Number of Pixels')
    plt.title(title)

    plt.plot(hist)

    plt.xlim([0, len(hist)])  # set x axis range
    plt.grid(True)
    plt.show()


def crop_partition(hist, img):
    lines = []
    for index, val in enumerate(hist):
        if val > 600:
            lines.append(index)

    width, height = img.shape[:2]
    wi = 170
    for i in range(1, 10):
        img_porte = img[wi * i:(wi * i + 100), 60:(width - 400)]
        cv2.imshow("img_crop", img_porte)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def extract_notes(binary_image):
    image_to_erode = cv2.bitwise_not(binary_image)
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(image_to_erode, kernel, iterations=3)
    # erosion = cv2.erode(image_to_erode, kernel, iterations=1)
    cv2.imshow('Erosion', erosion)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    dilatation = cv2.dilate(erosion, kernel, iterations=3)
    cv2.imshow('Dilatation', dilatation)
    # showImage(erosion)


def showImage(image):
    try:
        img = cv2.imread(image)
        binary = img_to_binary_grey_scale(img)
        hist = img_binary_vertical_hist(binary)
        display_hist(hist, "vertical histogram")
        cv2.imshow(image, binary)
    except cv2.error:
        print("Image file not found!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(image):
    #showImage(image)  # shows A SINGLE image in grayscale exercise 1
    # histogram(image)  # generates color histogram for image exercise 2
    img = cv2.imread(image)
    cv2.imshow('originale', img)
    binary = img_to_binary_grey_scale(img)
    hist = img_binary_vertical_hist(binary)
    # crop_partition(hist, img)
    extract_notes(binary)


if __name__ == '__main__':
    image = "images/partition.png"  # sys.argv[1]
    main(image)
    print('the end waiting for key interrupt...')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
