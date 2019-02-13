import numpy as np
import cv2
import time
import numpy as np
import os

protoFile = "/Users/tomcohen/Documents/ITC/project_2/openpose-master/models/pose/body_25/pose_deploy.prototxt"
weightsFile = "/Users/tomcohen/Documents/ITC/project_2/openpose-master/models/pose/body_25/pose_iter_584000.caffemodel"
nPoints = 25
POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
              [12, 13], [0, 14], [0, 15], [14, 16], [15, 17],
              [10, 11], [8, 12], [12, 13], [13, 14], [1, 0], [0, 15], [0, 16], [16, 18], [2, 17], [5, 18], [14, 19],
              [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]]
CURRENT_DIR = os.getcwd()
ORIGIN_DIR = os.path.join(CURRENT_DIR, './images/snaps/')
ESTIMATION_DIR = os.path.join(CURRENT_DIR, './images/estimation/')


def take_snapshots(origin_dir):
    """take n number of snapshots , finish using escape"""
    if not os.path.exists(origin_dir):
        os.makedirs(origin_dir)

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(origin_dir + img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()


def pose_estimation(file, origin_dir, estimation_dir):
    """
    :param file: name of the picture to estimate
    :return: convert one image to point and skeleton
    """
    image_name = file
    file = os.path.join(origin_dir, file)
    print(image_name, file)
    frame = cv2.imread(file)
    # frame = cv2.resize(frame, dsize=(1000, 800))
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    t = time.time()
    # input image dimensions for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    is_None = False
    df_is_empty = True
    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)
            is_None = True
        if is_None:
            continue
        flat_point = [e for l in points for e in l]
        #         print(flat_point)
        flat_array = np.array([e for l in points for e in l]) / 400
        point_dict = {i: flat_array[i] for i in np.arange(len(flat_array))}

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.imwrite(estimation_dir + 'points_' + image_name, frameCopy)
    cv2.imwrite(estimation_dir + 'skeleton_' + image_name, frame)

    print("Total time taken : {:.3f}".format(time.time() - t))

    cv2.waitKey(0)


def pose_estimation_all(origin_dir, estimation_dir):
    """
    :param origin_dir:
    :param estimation_dir:
    :return: takes the pictures taken in the previous steps and in a new folder saves these pictures with the point and skeleton
    """
    if not os.path.exists(estimation_dir):
        os.makedirs(estimation_dir)
    image_list = os.listdir(origin_dir)
    for image in image_list:
        pose_estimation(image, origin_dir, estimation_dir)


if __name__ == '__main__':
    take_snapshots(ORIGIN_DIR)
    pose_estimation_all(ORIGIN_DIR, ESTIMATION_DIR)
