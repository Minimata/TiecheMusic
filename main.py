import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
from operator import itemgetter


class Sheet:
    def __init__(self, line_list):
        self.lines = line_list


def img_to_binary_grey_scale(img, threshold):
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
    # cv2.imshow("Detected notes", im_with_keypoints)
    notes_positions = [point.pt for point in keypoints]
    return notes_positions


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


def crop_sheet_in_scope(img):
    hist = img_binary_vertical_hist(img)

    lines = [index for index, val in enumerate(hist) if val > 600]

    scope_first_line = list(lines[::5])
    number_scope = len(scope_first_line)

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
        if 5 <= val <= 15:
            hist_reduced.append((index, val))

    rising_edge = True
    rising_edge_list = []
    falling_edge_list = []
    for index, val in enumerate(hist_reduced):
        try:
            if hist_reduced[index + 1][1] > 5 and val[1] == 5 and rising_edge:
                rising_edge_list.append(val[0])
                rising_edge = False
            elif val[1] > 5 and hist_reduced[index +
                                             1][1] == 5 and not rising_edge:
                falling_edge_list.append(val[0])
                rising_edge = True
        except IndexError:
            pass

    note_list = []
    for i, val in enumerate(rising_edge_list):
        distance = falling_edge_list[i] - rising_edge_list[i]
        if distance < 8.0:
            middle_top = rising_edge_list[i] + math.ceil(distance / 2.0)
            x = middle_top - distance
            h_x = middle_top + distance
            note_list.append(line[0:height, x:h_x])
    return note_list


def detect_note(note_list):
    for note in note_list:
        hist = img_binary_vertical_hist(note)

        lines = []
        note = 0
        for index, val in enumerate(hist):
            if val > 10:
                lines.append(index)
            if val >= 3 and hist[index -
                                 1] >= 3 and hist[index +
                                                  1] >= 3 and hist[index -
                                                                   2] >= 3 and hist[index
                                                                                    +
                                                                                    2] >= 3:
                note = index

        try:
            if note < lines[0]:
                print("SOL 5", end=" ")
            elif note == lines[0]:
                print("FA 5", end=" ")
            elif lines[0] < note < lines[1]:
                print("MI 5", end=" ")
            elif note == lines[1]:
                print("RE 5", end=" ")
            elif lines[1] < note < lines[2]:
                print("DO 5", end=" ")
            elif note == lines[2]:
                print("SI 4", end=" ")
            elif lines[2] < note < lines[3]:
                print("LA 4", end=" ")
            elif note == lines[3]:
                print("SOL 4", end=" ")
            elif lines[3] < note < lines[4]:
                print("FA 4", end=" ")
            elif note == lines[4]:
                print("MI 4", end=" ")
            elif note > lines[4]:
                print("Re 4", end=" ")
        except IndexError:
            pass
        print("\t", end=" ")


def histogramm_process(binary):
    scope_list = crop_sheet_in_scope(binary)
    for line in scope_list:
        note_list = crop_line_in_note(line)
        detect_note(note_list)
        print()


def morpho_process(binary):
    notes_positions = extract_notes(binary)
    print(notes_positions)
    histo = img_binary_vertical_hist(binary)
    height, width = binary.shape
    lines_y = [i for i, count in enumerate(histo) if count > 0.5 * width]
    n = 5
    list_sheets = [lines_y[k:k + n] for k in range(0, len(lines_y), n)]
    sheets = [Sheet(l) for l in list_sheets]
    print("sorted note positions")
    notes_positions.sort(key=itemgetter(1, 0))
    print(notes_positions)
    # print(len(list_sheets))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def main():
    image = "images/partition.png"
    img = cv2.imread(image)
    binary_histo = img_to_binary_grey_scale(img, 20)
    binary_morpho = img_to_binary_grey_scale(img, 127)
    #choose the process
    # histogramm_process(binary_histo)
    morpho_process(binary_morpho)


if __name__ == '__main__':
    main()
