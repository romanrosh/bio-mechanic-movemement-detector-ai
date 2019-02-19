import cv2
import time
import numpy as np
import os
import pandas as pd

POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
              [12, 13], [0, 14], [0, 15], [14, 16], [15, 17],
              [10, 11], [8, 12], [12, 13], [13, 14], [1, 0], [0, 15], [0, 16], [16, 18], [2, 17], [5, 18], [14, 19],
              [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]]


inWidth = 368
inHeight = 368


def draw_sticks(path):
    df=pd.read_csv(path)
    df=df.loc[:,'0-XNose':"24-YRHeel"]
    # df.dropna(inplace=True,axis=0)
    # print(df)
    frame = np.zeros((800, 600))

    # vid_writer = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame.shape[1], frame.shape[0]))
    for j in range(len(df)):
        frame = np.zeros((800, 600))

        frame = cv2.resize(frame, dsize=(800, 600))
        tuples_array = []
        try:
            for i in range(0, 50, 2):
                tuples_array.append(tuple([int(df.iloc[j, i]), int(df.iloc[j, i+1])]))
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]
                # print(tuples_array)
                if tuples_array[partA] and tuples_array[partB]:
                    cv2.line(frame, tuples_array[partA], tuples_array[partB], (200, 200, 200), 2)
                    # cv2.circle(frame, tuples_array[partA], 8, (100, 100, 100))
                    # cv2.circle(frame, tuples_array[partB], 8, (100, 100, 100))
            cv2.putText(frame, "{}".format(str(df.iloc[j,-1])), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1,  #change to 1
                        lineType=cv2.LINE_AA)
        except Exception as ex:
            print(ex)
        print(frame.shape)
        from scipy import ndimage, misc
        misc.imsave('fileName.jpg', frame)
        frame = ndimage.imread('fileName.jpg', 0)
        frame = cv2.Canny(frame, 1, 100)

        cv2.imshow('Output-Keypoints', frame)
        cv2.imshow('Output-Skeleton', frame)
        # vid_writer.write(frame)
        cv2.waitKey(50)

# draw_sticks(path='C:/Users/romanrosh/PycharmProjects/bio-mechanic-movmement-detector-ai/destination/All_top_bottom_correct_movements.csv')
draw_sticks('C:/Users/romanrosh/PycharmProjects/bio-mechanic-movmement-detector-ai/destination/Data for fit/Most Squats in 5-Minutes-all labeled.csv')