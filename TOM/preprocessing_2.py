from PIL import Image
import pandas as pd
from preprocessing_1 import *
from scipy.signal import find_peaks

CURRENT_DIR = os.getcwd()
TRUTH = 1
VIDEO_DIR = os.path.join(CURRENT_DIR, './videos/' + str(TRUTH) + '/', )
TARGET_DIR = os.path.join(CURRENT_DIR, './videos_frames/' + str(TRUTH) + '/')
TARGET_CSV = os.path.join(CURRENT_DIR, './nps/')
ANCHOR = "0-YNose"

BODY_25_COLUMNS = ["0-XNose", "0-YNose",
                   "1-XNeck", "1-YNeck",
                   "2-XRShoulder", "2-YRShoulder",
                   "3-XRElbow", "3-YRElbow",
                   "4-XRWrist", "4-YRWrist",
                   "5-XLShoulder", "5-YLShoulder",
                   "6-XLElbow", "6-YLElbow",
                   "7-XLWrist", "7-YLWrist",
                   "8-XMidHip", "8-YMidHip",
                   "9-XRHip", "9-YRHip",
                   "10-XRKnee", "10-YRKnee",
                   "11-XRAnkle", "11-YRAnkle",
                   "12-XLHip", "12-YLHip",
                   "13-XLKnee", "13-YLKnee",
                   "14-XLA nkle", "14-YLAnkle",
                   "15-XREye", "15-YREye",
                   "16-XLEye", "16-YLEye",
                   "17-XREar", "17-YREar",
                   "18-XLEar", "18-YLEar",
                   "19-XLBigToe", "19-YLBigToe",
                   "20-XLSmallToe", "20-YLSmallToe",
                   "21-XLHeel", "21-YLHeel",
                   "22-XRBigToe", "22-YRBigToe",
                   "23-XRSmallToe", "23-YRSmallToe",
                   "24-XRHeel", "24-YRHeel"]

MODELLING_COLUMNS = ["0-XNose", "0-YNose",
                     "1-XNeck", "1-YNeck",
                     "2-XRShoulder", "2-YRShoulder",
                     "5-XLShoulder", "5-YLShoulder",
                     "8-XMidHip", "8-YMidHip",
                     "9-XRHip", "9-YRHip",
                     "10-XRKnee", "10-YRKnee",
                     "11-XRAnkle", "11-YRAnkle",
                     "13-XLKnee", "13-YLKnee",
                     "14-XLA nkle", "14-YLAnkle",
                     "19-XLBigToe", "19-YLBigToe",
                     "22-XRBigToe", "22-YRBigToe"]


def dir_to_body25(videos_dir, targets_dir):
    """convert all video from a directory into folder of frames and return a numpy array of all the body_25 coordinates
     of all the videos"""
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
    videos = os.listdir(videos_dir)
    df = pd.DataFrame()
    for i, video in enumerate(videos):
        # create new forlder for frames
        try:

            # creating a folder named data
            if not os.path.exists(targets_dir):
                os.makedirs(video)

                # if not created then raise error
        except OSError:
            print('Error: Creating directory of data')
            continue

        video_dir = videos_dir + video
        target_dir = targets_dir + video
        convertor = VideoToBody25(video_dir, target_dir)

        convertor.video_cut()

        try:
            array = pd.DataFrame(convertor.build_array())
            df = df.append(array)

        except NotADirectoryError:
            print('skipped: ' + str(video))
            continue
    logging.info('shape of df: {}'.format(df.shape))
    df.columns = BODY_25_COLUMNS
    return df


def split_preprocess(df, anchor, columns):
    """
    :param df: dataframe
    :param anchor: column to use to split the data
    :param columns: columns to keep
    :return: numpy array of array
    """
    df = df[columns]
    df = df.apply(lambda x: (x - x.max()) * (-1), axis=1)
    peaks, _ = find_peaks(df[anchor], distance=5)
    l = [0] + list(df.loc[peaks, '0-YNose'].index) + [len(df)]
    array = []
    for i in range(len(l)):
        if len(df[l[i - 1]:l[i]]) > 10:
            array.append(df[l[i - 1]:l[i]].values)
    array = np.array(array)
    return array

# def df_to_np(df):


if __name__ == '__main__':
    # video_to_img(os.path.join(CURRENT_DIR, TEST_FILE), os.path.join(CURRENT_DIR, TARGET_DIR))
    if not os.path.exists(TARGET_CSV):
        os.makedirs(TARGET_CSV)

    df = dir_to_body25(VIDEO_DIR, TARGET_DIR)
    array = split_preprocess(df, ANCHOR, MODELLING_COLUMNS)
    np.save(TARGET_CSV + str(TRUTH) + '.npy', array)
