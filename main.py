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


def crop_sheet_in_scope(hist, img):
    lines = []
    for index, val in enumerate(hist):
        if val > 600:
            lines.append(index)

    number_scope = 0
    scope_first_line = []
    for i, line in enumerate(lines):
        if i % 5 == 0:
            number_scope += 1
            scope_first_line.append(line)

    size_scope = math.fabs(lines[0] - lines[4])
    offset_scope = math.fabs(math.ceil(lines[4] - lines[5])/4)

    width, height = img.shape[:2]
    for i in range(0, number_scope):

        x = scope_first_line[i] - offset_scope
        h_x = size_scope + 2*offset_scope

        img_scope = img[x:x + h_x, 0: width]

        cv2.imshow("img_crop", img_scope)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_image(image):
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
    img = cv2.imread(image)
    binary = img_to_binary_grey_scale(img)
    hist = img_binary_vertical_hist(binary)
    crop_sheet_in_scope(hist, img)


if __name__ == '__main__':
    image = "images/partition.png"  # sys.argv[1]
    main(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
