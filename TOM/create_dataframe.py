from preprocessing_for_rnn import *
from PIL import Image
import pandas as pd

CURRENT_DIR = os.getcwd()
VIDEO_DIR = os.path.join(CURRENT_DIR, './videos/')
TARGET_DIR = os.path.join(CURRENT_DIR, './videos_frames/')
TARGET_CSV = os.path.join(CURRENT_DIR, './csv/')
TRUTH = 1

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
                   "14-XLAnkle", "14-YLAnkle",
                   "15-XREye", "15-YREye",
                   "16-XLEye", "16-YLEye",
                   "17-XREar", "17-YREar",
                   "18-XLEar", "18-YLEar",
                   "19-XLBigToe", "19-YLBigToe",
                   "20-XLSmallToe", "20-YLSmallToe",
                   "21-XLHeel", "21-YLHeel",
                   "22-XRBigToe", "22-YRBigToe",
                   "23-XRSmallToe", "23-YRSmallToe",
                   "24-XRHeel", "24-YRHeel", 'y']


def dir_to_body25(videos_dir, targets_dir, truth, rotation=None):
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
        convertor = VideoToBody25(video_dir, target_dir, truth)

        convertor.video_cut()

        # try:
        #     if rotation:
        #         print(rotation)
        #         list_images = os.listdir(target_dir)
        #         for image in list_images:
        #             image = os.path.join(target_dir, image)
        #             rotate_image(image, rotation)
        # except NotADirectoryError:
        #     print('skipped: ' + str(video))
        #     continue

        try:
            array = pd.DataFrame(convertor.build_array())
            df = df.append(array)

        except NotADirectoryError:
            print('skipped: ' + str(video))
            continue
    logging.info('shape of df: {}'.format(df.shape))
    df.columns = BODY_25_COLUMNS
    return df


if __name__ == '__main__':
    # video_to_img(os.path.join(CURRENT_DIR, TEST_FILE), os.path.join(CURRENT_DIR, TARGET_DIR))

    df = dir_to_body25(VIDEO_DIR, TARGET_DIR, TRUTH, rotation=270)
    df.to_csv(str(TRUTH) + '.csv')
