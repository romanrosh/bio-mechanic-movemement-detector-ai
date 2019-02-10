import cv2
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
TEST_FILE = 'IMG_3368.MOV'
TARGET_NAME = 'test'
CURRENT_DIR = os.getcwd()

protoFile = "/Users/tomcohen/Documents/ITC/project_2/openpose-master/models/pose/body_25/pose_deploy.prototxt"
weightsFile = "/Users/tomcohen/Documents/ITC/project_2/openpose-master/models/pose/body_25/pose_iter_584000.caffemodel"


def video_cut(filepath, img_destination):
    """
    :param filepath: path of the video
    :param img_destination: name of the folder where the images will be stored
    :return:
    """

    cam = cv2.VideoCapture(filepath)

    try:

        # creating a folder named data
        if not os.path.exists(img_destination):
            os.makedirs(img_destination)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

        # frame
    currentframe = 0

    while (True):

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = os.path.join(img_destination, str(currentframe) + '.jpg')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, dsize=(400, 400))
            # writing the extracted images
            cv2.imwrite(name, gray)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    logging.info('Converted: {}'.format(filepath.split('/')[-1]))
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def trace_skeleton(target_directory):
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    frame = cv2.imread("single.jpg")

    # Specify the input image dimensions
    inWidth = 368
    inHeight = 368
    nPoints = 25
    threshold = 0.1
    file_list = os.listdir(target_directory)
    print(file_list)

    the_trace = []

    for image_file in file_list:
        image = cv2.imread(os.path.join(target_directory, image_file))
        frameHeight, frameWidth, channels = image.shape

        inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]

        points = []
        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
            image_skel = np.zeros((frameHeight, frameWidth, 3))
            if prob > threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)

        the_trace.append(points)
        logging.info('traced: {}'.format(image_file))
    return the_trace


if __name__ == '__main__':
    # video_to_img(os.path.join(CURRENT_DIR, TEST_FILE), os.path.join(CURRENT_DIR, TARGET_DIR))
    TARGET_DIR = os.path.join(CURRENT_DIR, TARGET_NAME)


    video_cut(os.path.join(CURRENT_DIR, TEST_FILE), TARGET_DIR)
    # print(trace_skeleton(TARGET_DIR)[-1])