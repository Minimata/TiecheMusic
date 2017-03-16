
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np



def imgToBinaryGreyScale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # convert to grayscale
    ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    return thresh


def imgBinaryVerticalHist(img):
    height, width = img.shape
    hist = [0] * height

    s = 0
    while s < width * height:
        hist[math.floor(s/width)] += 1 - img.item(s) / 255
        s += 1

    return hist


def imgBinaryHorizontalHist(img):
    height, width = img.shape
    hist = [0] * width

    s = 0
    while s < width * height:
        hist[s % width] += 1 - img.item(s) / 255
        s += 1

    return hist


def displayHist(hist, title="histogram"):
    plt.xlabel('raw / column')
    plt.ylabel('Number of Pixels')
    plt.title(title)

    plt.plot(hist)

    plt.xlim([0, len(hist)])  # set x axis range
    plt.grid(True)
    plt.show()



def showImage(image):
    try:
        img = cv2.imread(image)
        binary = imgToBinaryGreyScale(img)
        hist = imgBinaryHorizontalHist(binary)
        displayHist(hist, "vertical histogram")
        cv2.imshow(image, binary)
    except cv2.error:
        print("Image file not found!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(image):
    showImage(image)  # shows A SINGLE image in grayscale exercise 1
    # histogram(image)  # generates color histogram for image exercise 2


if __name__ == '__main__':
    image = "../images/partition.png"  # sys.argv[1]
    main(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()