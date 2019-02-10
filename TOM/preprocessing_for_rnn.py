import cv2
import os
import numpy as np
import logging


logging.basicConfig(level=logging.INFO)
TEST_FILE = 'IMG_3368.MOV'
TARGET_NAME = 'test'
CURRENT_DIR = os.getcwd()
TARGET_DIR = os.path.join(CURRENT_DIR, TARGET_NAME)

protoFile = "/Users/tomcohen/Documents/ITC/project_2/openpose-master/models/pose/body_25/pose_deploy.prototxt"
weightsFile = "/Users/tomcohen/Documents/ITC/project_2/openpose-master/models/pose/body_25/pose_iter_584000.caffemodel"


class VideoToBody25:
    """object that will receive a path of a video and will eventually return a dataframe"""

    def __init__(self, video_path, img_destination):
        self.video_path = video_path
        self.img_destination = img_destination

    def video_cut(self):
        """
        :param filepath: path of the video
        :param img_destination: name of the folder where the images will be stored
        :return:
        """

        cam = cv2.VideoCapture(self.video_path)

        try:

            # creating a folder named data
            if not os.path.exists(self.img_destination):
                os.makedirs(self.img_destination)

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
                name = os.path.join(self.img_destination, str(currentframe) + '.jpg')

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, dsize=(400, 400))
                # writing the extracted images
                cv2.imwrite(name, gray)

                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
            else:
                break

        logging.info('Converted: {}'.format(self.video_path.split('/')[-1]))
        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

    def trace_skeleton(self, image_path):
        """
        :param image_path: path of a specific image
        :return: coordinates recieved when we feed forward
        """

        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

        # Specify the input image dimensions
        inWidth = 368
        inHeight = 368
        nPoints = 25
        threshold = 0.1
        image = cv2.imread(image_path)
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

        return points

    def build_array(self):
        """

        :return: numpy array with the first array representing the columns and the rest the coordinates for each frame
        in the video
        """
        self.video_cut()
        array = np.array(list(range(25)))
        file_list = sorted(os.listdir(self.img_destination), key=lambda file: int(file.split('.')[0]))

        for image_name in file_list:
            image_path = os.path.join(self.img_destination, image_name)
            frame_coord = self.trace_skeleton(image_path)
            array = np.vstack([array, frame_coord])

            logging.info('traced: {}'.format(image_name))
        return array


if __name__ == '__main__':
    # video_to_img(os.path.join(CURRENT_DIR, TEST_FILE), os.path.join(CURRENT_DIR, TARGET_DIR))

    video_path = os.path.join(CURRENT_DIR, TEST_FILE)
    converter = VideoToBody25(video_path, TARGET_DIR)

    print(converter.build_array())
