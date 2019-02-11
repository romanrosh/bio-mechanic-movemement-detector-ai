from preprocessing_for_rnn import *
from PIL import Image

CURRENT_DIR = os.getcwd()
VIDEO_DIR = os.path.join(CURRENT_DIR, './videos')
TARGET_DIR = os.path.join(CURRENT_DIR, './videos_frames')


def rotate_image(img_file, degrees):
    """
    :param img_file: image path to rotate
    :param degrees:  degrees to rotate
    """
    im1 = Image.open(img_file)
    im2 = im1.rotate(degrees)
    im2.save(img_file)


def dir_to_body25(VIDEO_DIR, TARGET_DIR):
    """convert all video from a directory into folder of frames and return a numpy array of all the body_25 coordinates
     of all the videos"""
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
    videos = os.listdir(VIDEO_DIR)
    array = np.zeros((len(videos)), dtype=object)
    for i, video in enumerate(videos):
        # create new forlder for frames
        try:

            # creating a folder named data
            if not os.path.exists(TARGET_DIR):
                os.makedirs(video)

                # if not created then raise error
        except OSError:
            print('Error: Creating directory of data')
        video_dir = VIDEO_DIR + '/' + video
        target_dir = TARGET_DIR + '/' + video + '/'
        convertor = VideoToBody25(video_dir, target_dir)

        convertor.video_cut()
        array[i] = convertor.build_array()

    return array


if __name__ == '__main__':
    # video_to_img(os.path.join(CURRENT_DIR, TEST_FILE), os.path.join(CURRENT_DIR, TARGET_DIR))

    array = dir_to_body25(VIDEO_DIR, TARGET_DIR)
    np.save('fnumpy.npy', array)
