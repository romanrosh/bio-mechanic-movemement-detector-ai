from preprocessing_for_rnn import *
from PIL import Image
import pandas as pd
from argparse import Namespace

CURRENT_DIR = os.getcwd()
VIDEO_DIR = os.path.join(CURRENT_DIR, './videos/')
TARGET_DIR = os.path.join(CURRENT_DIR, './videos_frames/')
TARGET_CSV = os.path.join(CURRENT_DIR, './csv/')
TRUTH = 1



def rotate_image(img_file, degrees):
    """
    :param img_file: image path to rotate
    :param degrees:  degrees to rotate
    """
    im1 = Image.open(img_file)
    im2 = im1.rotate(degrees)
    im2.save(img_file)


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
    return df


if __name__ == '__main__':
    # video_to_img(os.path.join(CURRENT_DIR, TEST_FILE), os.path.join(CURRENT_DIR, TARGET_DIR))

    df = dir_to_body25(VIDEO_DIR, TARGET_DIR, TRUTH, rotation=270)
    df.to_csv(str(TRUTH) + '.csv')
