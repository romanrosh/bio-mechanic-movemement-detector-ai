from preprocessing_for_rnn import *
from PIL import Image

CURRENT_DIR = os.getcwd()
VIDEO_DIR = os.path.join(CURRENT_DIR, './videos/')
TARGET_DIR = os.path.join(CURRENT_DIR, './videos_frames/')


def rotate_image(img_file, degrees):
    """
    :param img_file: image path to rotate
    :param degrees:  degrees to rotate
    """
    im1 = Image.open(img_file)
    im2 = im1.rotate(degrees)
    im2.save(img_file)


def dir_to_body25(videos_dir, targets_dir, rotation = None):
    """convert all video from a directory into folder of frames and return a numpy array of all the body_25 coordinates
     of all the videos"""
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
    videos = os.listdir(videos_dir)
    array = np.zeros((len(videos)), dtype=object)
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
            if rotation:
                print(rotation)
                list_images = os.listdir(target_dir)
                for image in list_images:
                    image = os.path.join(target_dir, image)
                    rotate_image(image, rotation)
        except NotADirectoryError:
            print('skipped: ' + str(video))
            continue

        try:
            array[i] = convertor.build_array()

        except NotADirectoryError:
            print('skipped: ' + str(video))
            continue

    return array


if __name__ == '__main__':
    # video_to_img(os.path.join(CURRENT_DIR, TEST_FILE), os.path.join(CURRENT_DIR, TARGET_DIR))

    array = dir_to_body25(VIDEO_DIR, TARGET_DIR, rotation=270)
    np.save('fnumpy.npy', array)
