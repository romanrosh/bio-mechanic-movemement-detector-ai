import cv2
import time
import numpy as np
import pandas as pd
import os
from angles import angle

MODE = "BODY25"
input_source = "250 consecutive squats.mp4"
# 100 Body Squats Non Stop.mp4
# --
# Most Squats in 5-Minutes.mp4
# The Air Squat.mp4
# The Back Squat.mp4
# Roman10.MOV
# 250 consecutive squats.mp4

output_destination ='./destination/' + input_source.split('.')[0] + '.avi'
OUTPUT_CSV = './destination/output.csv'
FRAMES_TO_TAKE = 30

if MODE is "COCO":
    protoFile = "C:/Users/romanrosh/openpose-1.4.0-win64-gpu-binaries/models/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "C:/Users/romanrosh/openpose-1.4.0-win64-gpu-binaries/models/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

elif MODE is "MPI":
    protoFile = "C:/Users/romanrosh/openpose-1.4.0-win64-gpu-binaries/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "C:/Users/romanrosh/openpose-1.4.0-win64-gpu-binaries/models/pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                  [11, 12], [12, 13]]

elif MODE is "BODY25":
    protoFile = "C:/Users/romanrosh/openpose-1.4.0-win64-gpu-binaries/models/pose/body_25/pose_deploy.prototxt"
    weightsFile = "C:/Users/romanrosh/openpose-1.4.0-win64-gpu-binaries/models/pose/body_25/pose_iter_584000.caffemodel"
    nPoints = 25
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17],
                  [10, 11], [8, 12], [12, 13], [13, 14], [1, 0], [0, 15], [0, 16], [16, 18], [2, 17], [5, 18], [14, 19],
                  [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]]

    BODY_25_COLUMNS = ["0-XNose",       "0-YNose",
                       "1-XNeck",       "1-YNeck",
                       "2-XRShoulder",  "2-YRShoulder",
                       "3-XRElbow",     "3-YRElbow",
                       "4-XRWrist",     "4-YRWrist",
                       "5-XLShoulder",  "5-YLShoulder",
                       "6-XLElbow",     "6-YLElbow",
                       "7-XLWrist",     "7-YLWrist",
                       "8-XMidHip",     "8-YMidHip",
                       "9-XRHip",       "9-YRHip",
                       "10-XRKnee",     "10-YRKnee",
                       "11-XRAnkle",    "11-YRAnkle",
                       "12-XLHip",      "12-YLHip",
                       "13-XLKnee",     "13-YLKnee",
                       "14-XLAnkle",    "14-YLAnkle",
                       "15-XREye",      "15-YREye",
                       "16-XLEye",      "16-YLEye",
                       "17-XREar",      "17-YREar",
                       "18-XLEar",      "18-YLEar",
                       "19-XLBigToe",   "19-YLBigToe",
                       "20-XLSmallToe", "20-YLSmallToe",
                       "21-XLHeel",     "21-YLHeel",
                       "22-XRBigToe",   "22-YRBigToe",
                       "23-XRSmallToe", "23-YRSmallToe",
                       "24-XRHeel",     "24-YRHeel"]

inWidth = 368
inHeight = 368
threshold = 0.1

cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

vid_writer = cv2.VideoWriter(output_destination, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                             (frame.shape[1], frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
counter = 0
df_is_empty = True
while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    counter += 1
    if np.mod(counter, FRAMES_TO_TAKE) != 0:
        continue
    print('frame', counter)
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

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
    flat_array = []
    for point in points:
        if point is None:
            flat_array.append(None)
            flat_array.append(None)
        else:
            for feature in point:
                flat_array.append(feature)
    flat_array = pd.Series(np.array(flat_array))
    # flat_array = np.array([feature for point in points for feature in point])
    if df_is_empty:
        df = pd.DataFrame([flat_array])
        df_is_empty = False
    else:
        df = df.append(flat_array, ignore_index=True)
    print('dataframe size', len(df))
    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        # print(points)
    # cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,
    #             (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)
    vid_writer.write(frame)
    if hasFrame == False:
        break

vid_writer.release()

## add column names to the dataframe
df.columns = BODY_25_COLUMNS
# hip_vector = []
# shin_vector = []
# knee_angle = []
# for row in df.iterrows():
#     print(row)
#     try:
#         hip_vector.append(
#             np.array([row['right hip x'] - row['right knee x'], row['right hip y'] - row['right knee y']]))
#         shin_vector.append(
#             np.array([row['right ankle x'] - row['right knee x'], row['right ankle y'] - row['right knee y']]))
#         knee_angle.append(angle(hip_vector[-1], shin_vector[-1]))
#     except:
#         print('error in angle calculation')
#         hip_vector.append(None)
#         shin_vector.append(None)
#         knee_angle.append(None)
#
# df['hip vector'] = pd.Series(hip_vector)
# df['knee vector'] = pd.Series(shin_vector)
# df['knee angle'] = pd.Series(knee_angle)

exists = os.path.isfile(OUTPUT_CSV)
if exists:
    with open(OUTPUT_CSV, 'a') as f:
        df.to_csv(f, header=False)
else:
    df.to_csv(OUTPUT_CSV)
