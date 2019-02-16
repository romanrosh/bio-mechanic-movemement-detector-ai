import cv2
import time
import numpy as np
import os
import pandas as pd

MODE = "BODY25"
OUTPUT_CSV='output.csv'
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

def read_from_folder(path):
    files = os.listdir(path)
    df_is_empty = True
    # df.loc[j, 'Source-Image'] = path.split('/')[-2] + '/' + name
    file_list = [path.split('/')[-3] + '/' + path.split('/')[-2] + '/' + file for file in files]
    for j,name in enumerate(files):
        print(name)
        frame = cv2.imread(path+name)
        frame = cv2.resize(frame, dsize=(800, 600))
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
        # print("time taken by network : {:.3f}".format(time.time() - t))

        H = output.shape[2]
        W = output.shape[3]

        # Empty list to store the detected keypoints
        points = []

        is_None = False
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
        flat_array=[]
        for point in points:
            if point is None:
                flat_array.append(None)
                flat_array.append(None)
            else:
                for feature in point:
                    flat_array.append(feature)
            point_dict = {i: flat_array[i] for i in np.arange(len(flat_array))}

        flat_array = pd.Series(np.array(flat_array))
        if df_is_empty:
            df = pd.DataFrame([flat_array])
            df_is_empty = False
        else:
            df = df.append([flat_array], ignore_index=True)
        # df.columns = BODY_25_COLUMNS
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
                cv2.circle(frame, points[partA], 8, (0, 0, 255))

        cv2.imshow('Output-Keypoints', frameCopy)
        cv2.imshow('Output-Skeleton', frame)

        # cv2.imwrite('./destination/'+name+frameCopy, path+name+frameCopy)
        # cv2.imwrite('./destination/'+path+name+frame, path+name+frame)

        # print("Total time taken : {:.3f}".format(time.time() - t))
        cv2.waitKey(0)
    df.columns = BODY_25_COLUMNS
    df['Source-Image'] = file_list

    for i in range(len(df)):
        # RIGHT KNEE
        u = (df.loc[i, '9-XRHip'] - df.loc[i, '10-XRKnee'], df.loc[i, '9-YRHip'] - df.loc[i, '10-YRKnee'])
        v = (df.loc[i, '11-XRAnkle'] - df.loc[i, '10-XRKnee'], df.loc[i, '11-YRAnkle'] - df.loc[i, '10-YRKnee'])
        c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
        df.loc[i,'RightKneeAngle'] = np.arccos(np.clip(c, -1, 1))*180/np.pi
        # LEFT KNEE
        u = (df.loc[i, '12-XLHip'] - df.loc[i, '13-XLKnee'], df.loc[i, '12-YLHip'] - df.loc[i, '13-YLKnee'])
        v = (df.loc[i, '14-XLAnkle'] - df.loc[i, '13-XLKnee'], df.loc[i, '14-YLAnkle'] - df.loc[i, '13-YLKnee'])
        c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
        df.loc[i,'LeftKneeAngle'] = np.arccos(np.clip(c, -1, 1))*180/np.pi
        # heel ankle toe knee left side
        toes_x = (df.loc[i, "19-XLBigToe"] + df.loc[i, "20-XLSmallToe"]) / 2
        toes_y = (df.loc[i, "19-YLBigToe"] + df.loc[i, "20-YLSmallToe"]) / 2
        heel_angle_x = (df.loc[i, "21-XLHeel"] + df.loc[i, "14-XLAnkle"]) / 2
        heel_angle_y = (df.loc[i, "21-YLHeel"] + df.loc[i, "14-XLAnkle"]) / 2
        u = (toes_x - heel_angle_x, toes_y - heel_angle_y)
        v = (heel_angle_x - df.loc[i, '13-XLKnee'], heel_angle_y - df.loc[i, '13-YLKnee'])
        c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
        df.loc[i,'LeftHeelAngleAngle'] = np.arccos(np.clip(c, -1, 1))*180/np.pi
        # heel ankle toe knee right side
        toes_x = (df.loc[i, "22-XRBigToe"] + df.loc[i, "23-XRSmallToe"]) / 2
        toes_y = (df.loc[i, "22-YRBigToe"] + df.loc[i, "23-YRSmallToe"]) / 2
        heel_angle_x = (df.loc[i, "24-XRHeel"] + df.loc[i, "11-XRAnkle"]) / 2
        heel_angle_y = (df.loc[i, "24-YRHeel"] + df.loc[i, "11-YRAnkle"]) / 2
        u = (toes_x - heel_angle_x, toes_y - heel_angle_y)
        v = (heel_angle_x - df.loc[i, '10-XRKnee'], heel_angle_y - df.loc[i, '10-YRKnee'])
        c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
        df.loc[i,'RightHeelAngleAngle'] = np.arccos(np.clip(c, -1, 1))*180/np.pi

        # hip neck knee
        knee_x = (df.loc[i, "10-XRKnee"] + df.loc[i, "13-XLKnee"]) / 2
        knee_y = (df.loc[i, "10-YRKnee"] + df.loc[i, "13-YLKnee"]) / 2
        u = (knee_x - df.loc[i, '8-XMidHip'], knee_y - df.loc[i, '8-YMidHip'])
        v = (df.loc[i, '1-XNeck'] - df.loc[i, '8-XMidHip'], df.loc[i, '1-YNeck'] - df.loc[i, '8-YMidHip'])
        c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
        df.loc[i,'HipAngle'] = np.arccos(np.clip(c, -1, 1))*180/np.pi

    exists = os.path.isfile('./destination/'+OUTPUT_CSV)

    if exists:
        with open('./destination/'+OUTPUT_CSV, 'a') as f:
            df.to_csv(f, header=False)
    else:
        df.to_csv('./destination/'+OUTPUT_CSV)
    return df

folders = ['C:/Users/romanrosh/photos/front/right_bottom/',
        'C:/Users/romanrosh/photos/front/right_top/',
        'C:/Users/romanrosh/photos/side/right_bottom/',
        'C:/Users/romanrosh/photos/side/right_top/']

# folders = ['C:/Users/romanrosh/photos/front/right_bottom/',
#         'C:/Users/romanrosh/photos/front/right_top/']

folders = ['C:/Users/romanrosh/photos/test/']

for folder in folders:
    read_from_folder(folder)