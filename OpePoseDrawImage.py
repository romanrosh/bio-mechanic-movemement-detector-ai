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
    df.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1,inplace=True)
    frame = np.zeros((300, 300 * 3))
    frame = cv2.resize(frame, dsize=(300, 300 * 3))
    vid_writer = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame.shape[1], frame.shape[0]))
    for j in range(len(df)):
        # frame = np.zeros((300,300*3))
        # frame = cv2.resize(frame, dsize=(300, 300*3))
        tuples_array = []
        try:
            for i in range(0, 50, 2):
                tuples_array.append(tuple([int(df.iloc[j, i]), int(df.iloc[j, i+1])]))
            # print(tuples_array)
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]
                if tuples_array[partA] and tuples_array[partB]:
                    cv2.line(frame, tuples_array[partA], tuples_array[partB], (100, 100, 100), 2)
                    cv2.circle(frame, tuples_array[partA], 8, (100, 100, 100))
                    cv2.circle(frame, tuples_array[partB], 8, (100, 100, 100))
        except Exception:
            print('Error in image generation')
            # print(points)
        # cv2.imshow('Output-Keypoints', frame)
        cv2.imshow('Output-Skeleton', frame)
        vid_writer.write(frame)
        cv2.waitKey(0)


draw_sticks(path='C:/Users/romanrosh/PycharmProjects/bio-mechanic-movmement-detector-ai/destination/all labeled.csv')