
import cv2
import matplotlib.pyplot as plt
import sys


def histogram(image):
    # initialize histogram infographics
    plt.xlabel('Color Value [0 - 255]')
    plt.ylabel('Number of Pixels')
    plt.title('Color appearances')
    # initialize image information
    try:
        img = cv2.imread(image)
        # set image type to RGB instead of BGR(default)
        rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except cv2.error:
        print("Image file not found!")

    # initialize histogram data using auto-generated (img.ravel()) data
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        # generates histogram for each color r g b
        histr = cv2.calcHist([rgbimg], [i], None, [256], [0, 256])
        print(i, col)
        plt.plot(histr, color=col)
    plt.xlim([0, 256])  # set x axis range
    plt.grid(True)
    plt.show()


def showImage(image):
    try:
        img = cv2.imread(image)
        # im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #convert to grayscale
        cv2.imshow(image, img)
    except cv2.error:
        print("Image file not found!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(image):
    showImage(image)  # shows A SINGLE image in grayscale exercise 1
    histogram(image)  # generates color histogram for image exercise 2


if __name__ == '__main__':
    image = "lena.png"  # sys.argv[1]
    main(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
