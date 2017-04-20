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

    size_scope = int(abs(lines[0] - lines[4]))
    offset_scope = int(abs(math.ceil(lines[4] - lines[5]) / 4))
    width, height = img.shape[:2]

    for i in range(0, number_scope):

        x = scope_first_line[i] - offset_scope
        h_x = size_scope + 2 * offset_scope

        img_scope = img[x:x + h_x, 0:width]

        cv2.imshow("img_crop", img_scope)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def extract_notes(binary_image):
    # cv2.imshow('originale', binary_image)
    image_to_erode = cv2.bitwise_not(binary_image)
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(image_to_erode, kernel, iterations=3)
    dilatation = cv2.dilate(erosion, kernel, iterations=7)
    extracted_notes_image = cv2.bitwise_not(dilatation)
    # cv2.imshow('Morpho result', extracted_notes_image)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 90

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.6

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    # keypoints = detector.detect(binary_image)
    keypoints = detector.detect(extracted_notes_image)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(
        extracted_notes_image, keypoints,
        np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Detected notes", im_with_keypoints)
    notes_positions = [point.pt for point in keypoints]
    return notes_positions


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


def histogramm_process(binary):
    hist = img_binary_vertical_hist(binary)
    crop_sheet_in_scope(hist, binary)


def morpho_process(binary):
    extract_notes(binary)


def main():
    image = "images/partition.png"  # sys.argv[1]
    img = cv2.imread(image)
    binary = img_to_binary_grey_scale(img)
    #choose the process
    #histogramm_process(binary)
    morpho_process(binary)


if __name__ == '__main__':
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
