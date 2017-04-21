import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


def img_to_binary_grey_scale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # convert to grayscale
    ret, thresh = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
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


def show_hist(hist, title="histogram"):
    plt.xlabel('raw / column')
    plt.ylabel('Number of Pixels')
    plt.title(title)

    plt.plot(hist)

    plt.xlim([0, len(hist)])  # set x axis range
    plt.grid(True)
    plt.show()


def crop_sheet_in_scope(img):
    hist = img_binary_vertical_hist(img)
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

    size_scope = int(abs(lines[0] - lines[4]))
    offset_scope = int(abs(math.ceil(lines[4] - lines[5]) / 6))
    width, height = img.shape[:2]

    scope_list = []
    for i in range(0, number_scope):

        x = scope_first_line[i] - offset_scope
        h_x = size_scope + 2 * offset_scope

        scope_list.append(img[x:x + h_x, 0:width])

    return scope_list


def crop_line_in_note(line):
    width, height = line.shape[:2]
    hist = img_binary_horizontal_hist(line)
    hist_reduced = []
    for index, val in enumerate(hist):
        if val >= 5 and val <= 15:
            hist_reduced.append((index,val))

    rising_edge = True
    rising_edge_list = []
    falling_edge_list = []
    for index, val in enumerate(hist_reduced):
        try:
            if hist_reduced[index + 1][1] > 5.0 and val[1] == 5.0 and rising_edge:
                rising_edge_list.append(val[0])
                rising_edge = False
            elif val[1] > 5.0 and hist_reduced[index + 1][1] == 5.0 and not rising_edge:
                falling_edge_list.append(val[0])
                rising_edge = True
        except IndexError:
            pass

    note_list = []
    for i in range(0, len(rising_edge_list)):
        distance = falling_edge_list[i] - rising_edge_list[i]
        if distance < 8.0:
            middle_top = rising_edge_list[i]+ math.ceil(distance/2.0)
            x = middle_top - distance
            h_x = middle_top + distance
            note_list.append(line[0:height, x:h_x])
    return note_list

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


def show_image(image):
    try:
        img = cv2.imread(image)
        binary = img_to_binary_grey_scale(img)
        hist = img_binary_vertical_hist(binary)
        show_hist(hist, "vertical histogram")
        cv2.imshow(image, binary)
    except cv2.error:
        print("Image file not found!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_note(note_list):
    for i in range(0, len(note_list)):
        cv2.imshow("couco", note_list[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def histogramm_process(binary):
    scope_list = crop_sheet_in_scope(binary)
    for line in scope_list:
        note_list = crop_line_in_note(line)
        print_note(note_list)


def morpho_process(binary):
    pass


def main():
    image = "images/partition.png"  # sys.argv[1]
    img = cv2.imread(image)
    binary = img_to_binary_grey_scale(img)
    #choose the process
    histogramm_process(binary)

    #morpho_process(binary)


if __name__ == '__main__':
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
