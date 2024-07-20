import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
os.chdir(r"D:\Github\AutoSpot")


def read_img(path):
    img = cv.imread(path)
    return img


def rescale_img(img, scale):
    # images, videos, live video
    dimensions = int(img.shape[1] * scale), int(img.shape[0] * scale)  # [1] horizontal
    if scale < 1:
        return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    else:
        return cv.resize(img, dimensions, interpolation=cv.INTER_CUBIC)


def change_res(capture, width, height):
    # live video
    capture.set(3, width)
    capture.set(4, height)


def grayscale(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray


def blur(img, degree):
    blurred = cv.GaussianBlur(img, (degree, degree), cv.BORDER_DEFAULT)
    return blurred


def edge(img, thresh_low, thresh_high):
    edged = cv.Canny(img, thresh_low, thresh_high)  # have threshold values
    return edged


def edge_expand(img, size, degree):
    dilated = cv.dilate(img, (size, size), iterations=degree)
    return dilated


def edge_shrink(img,size, degree):
    shrunk = cv.erode(img, (size, size), iterations=degree)
    return shrunk


def crop(img, x_range, y_range):
    cropped = img[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    return cropped


def translate(img, x, y):
    trans_mat = np.float32([[1, 0, x], [0, 1, y]])  # create a 2x3 matrix
    dimensions = img.shape[1], img.shape[0]
    return cv.warpAffine(img, trans_mat, dimensions)  # use the matrix to modify the img


def rotate(img, angle, center=None):
    dim = img.shape[:2]
    if center is None:
        center = (dim[1]//2, dim[0]//2)  # // is integer div, 1 is for width in array
    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)  # make a rotation matrix
    return cv.warpAffine(img, rot_mat, dim)  # use the matrix to modify the img


def flip(img, setting):
    return cv.flip(img, setting)  # 0 vertical, 1 horizontal, -1 both


def read_video():
    path = ""
    capture = cv.VideoCapture(path)  # can be used to read webcams

    while True:  # read every frame
        readable, frame = capture.read()  # returns (can be read), (the frame)
        cv.imshow('Video', frame)

        if cv.waitKey(20) & 0xFF == ord('d'):  # pressing d will stop it
            break
    capture.release()
    cv.destroyAllWindows()


def contour_detect(img):
    a = grayscale(img)
    b = blur(a, 3)
    c = edge(b, 125, 175)
    cv.imshow("edges", c)
    # chain approx simple removes unnecessary points
    contour_pos, contour_layer = cv.findContours(c, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(c.shape, dtype='uint8')
    cv.drawContours(blank, contour_pos, -1, (255, 255, 255), 1)  # draw it on the blank
    cv.imshow("contours", blank)


def histogram(img):
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title("Histogram")
    plt.xlabel('bins')
    plt.xlabel('pixels')
    for i, col in enumerate(colors):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()


def adaptive_thresh(img):
    a = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 13, 3)
    cv.imshow("adaptive thresh", a)
    return a


def face_detection_haar():
    images = []
    faces_rects = []
    for img_file in glob.glob(os.path.join("using/", '*.png')):
        with open(img_file, 'r', encoding="utf8") as f:
            images.append(read_img(img_file))
    # create cascade object
    haar_cascade = cv.CascadeClassifier('../haar_face.xml')

    for i, image in enumerate(images):
        # return a list of rectangles marking face(s)
        to_read = rescale_img(grayscale(image), 1)
        start = time.time()
        faces_rect = haar_cascade.detectMultiScale(to_read, scaleFactor=1.1, minNeighbors=1)
        print(time.time()-start)
        faces_rects.append(faces_rect)
        for (x, y, w, h) in faces_rect:
            cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
        cv.imshow(str(i), image)


def face_recognizer():
    images = []
    faces_rects = []
    for img_file in glob.glob(os.path.join("using/", '*.png')):
        with open(img_file, 'r', encoding="utf8") as f:
            images.append(read_img(img_file))
    # create cascade object
    haar_cascade = cv.CascadeClassifier('../haar_face.xml')

    for i, image in enumerate(images):
        # return a list of rectangles marking face(s)
        to_read = rescale_img(grayscale(image), 1)
        start = time.time()
        faces_rect = haar_cascade.detectMultiScale(to_read, scaleFactor=1.1, minNeighbors=1)
        print(time.time()-start)
        faces_rects.append(faces_rect)
        for (x, y, w, h) in faces_rect:
            cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
        cv.imshow(str(i), image)

# b, g, r = cv.split(img) splits into the components
# cv.merge([b, g, r]) merges the channels


face_detection_haar()
cv.waitKey(0)
