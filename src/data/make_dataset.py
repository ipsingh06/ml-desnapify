# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import shutil
import cv2
import dlib
import math
from imutils import face_utils, rotate_bound
import matplotlib.pyplot as plt
from enum import Enum
from tqdm import tqdm
import numpy as np
import h5py
import parmap
from collections import OrderedDict
import random
import hashlib


MIN_FACE_SIZE = 200

DB_SPLIT_TRAIN = 0.6
DB_SPLIT_VALIDATION = 0.2

INTERIM_ORIG_DIR = "orig"
INTERIM_TRANSFORMED_DIR = "transformed"


class FaceDetector:
    def __init__(self, caascades_path: str):
        self.__detector = dlib.get_frontal_face_detector()
        # link to model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        model = os.path.join(caascades_path, 'shape_predictor_68_face_landmarks.dat')
        self.__predictor = dlib.shape_predictor(model)

    def has_face(self, img, check_size=False) -> bool:
        # convert BGR image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(cv_rgb)
        # plt.show()

        faces = self.__detector(gray, 0)
        if len(faces) != 1:
            return False
        face = faces[0]
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        return not check_size or (h > MIN_FACE_SIZE and w > MIN_FACE_SIZE)

    def get_landmarks(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.__detector(gray, 0)
        shape = self.__predictor(gray, faces[0])
        shape = face_utils.shape_to_np(shape)
        return faces[0], shape


class ImageProcessor:
    class Filter(Enum):
        ORIGINAL = 1
        DOG = 2

    def __init__(self, output_width: int, output_height: int, sprites_path: str):
        self.__output_width = output_width
        self.__output_height = output_height
        self.__sprites_path = sprites_path

    def resize(self, img):
        # resize the image
        height, width, _ = img.shape
        scale_height = self.__output_height / height
        scale_width = self.__output_width / width
        if scale_height == scale_width:
            resize_height = self.__output_height
            resize_width = self.__output_width
        elif scale_height > scale_width:
            resize_height = self.__output_height
            resize_width = int(width * scale_height)
        else:
            resize_width = self.__output_width
            resize_height = int(height * scale_width)
        img = cv2.resize(img, (resize_width, resize_height))
        # crop image
        if resize_width > self.__output_width:
            x = (resize_width-self.__output_width)//2
            img = img[:, x:x+self.__output_width]
        elif resize_height > self.__output_height:
            y = (resize_height-self.__output_height)//2
            img = img[y:y+self.__output_height, :]
        return img

    def process(self, img, face, landmarks, output_filter: Filter):
        img = img.copy()
        if output_filter == ImageProcessor.Filter.DOG:
            return self.__process_dog(img, face, landmarks)
        else:
            raise NotImplementedError("Unsupported filter {}".format(output_filter))

    def __process_dog(self, img, face, landmarks):
        def adjust_bounding_box(_x, _y, _w, _h, factor):
            factor_x, factor_y = factor
            center_x, center_y = _x+int(_w/2.0), _y+int(_h/2.0)
            _w, _h = int(factor_x*_w), int(factor_y*_h)
            _x, _y = center_x-int(_w/2.0), center_y-int(_h/2.0)
            return _x, _y, _w, _h

        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        # inclination based on eyebrows
        incl = ImageProcessor.__calculate_inclination(landmarks[17], landmarks[26])
        # y coordiantes of landmark points of lips
        is_mouth_open = (landmarks[66][1] - landmarks[62][1]) >= 10
        (x0, y0, w0, h0) = ImageProcessor.__get_face_boundbox(landmarks, face, 6)  # bound box of mouth
        (x3, y3, w3, h3) = ImageProcessor.__get_face_boundbox(landmarks, face, 5)  # nose
        (x4, y4, w4, h4) = ImageProcessor.__get_face_boundbox(landmarks, face, 8)  # top of head

        # expand the nose bounding box to adjust the nose
        x3, y3, w3, h3 = adjust_bounding_box(x3, y3, w3, h3, (2.7, 1.5))
        # expand the face bounding box to adjust the ears
        x4, y4, w4, h4 = adjust_bounding_box(x4, y4, w4, h4, (1.7, 1.0))

        self.__apply_sprite(img, 'doggy_nose.png', w3, x3, y3, incl, ontop=False)

        self.__apply_sprite(img, 'doggy_ears.png', w4, x4, y4, incl, ontop=False)

        if is_mouth_open:
            self.__apply_sprite(img, 'doggy_tongue.png', w0, x0, y0, incl, ontop=False)
        return img

    @staticmethod
    def __calculate_inclination(point1, point2):
        x1,x2,y1,y2 = point1[0], point2[0], point1[1], point2[1]
        incl = 180/math.pi*math.atan((float(y2-y1))/(x2-x1))
        return incl

    def __apply_sprite(self, image, path2sprite, w, x, y, angle, ontop = True):
        path2sprite = os.path.join(self.__sprites_path, path2sprite)
        sprite = cv2.imread(path2sprite,-1)
        #print sprite.shape
        sprite = rotate_bound(sprite, angle)
        (sprite, y_final) = ImageProcessor.__adjust_sprite2head(sprite, w, y, ontop)
        image = ImageProcessor.__draw_sprite(image,sprite,x, y_final)

    # Draws sprite over a image
    # It uses the alpha chanel to see which pixels need to be reeplaced
    # Input: image, sprite: numpy arrays
    # output: resulting merged image
    @staticmethod
    def __draw_sprite(frame, sprite, x_offset, y_offset):
        (h,w) = (sprite.shape[0], sprite.shape[1])
        (imgH,imgW) = (frame.shape[0], frame.shape[1])

        if y_offset+h >= imgH: #if sprite gets out of image in the bottom
            sprite = sprite[0:imgH-y_offset,:,:]

        if x_offset+w >= imgW: #if sprite gets out of image to the right
            sprite = sprite[:,0:imgW-x_offset,:]

        if x_offset < 0: #if sprite gets out of image to the left
            sprite = sprite[:,abs(x_offset)::,:]
            w = sprite.shape[1]
            x_offset = 0

        #for each RGB chanel
        for c in range(3):
                #chanel 4 is alpha: 255 is not transpartne, 0 is transparent background
                frame[y_offset:y_offset+h, x_offset:x_offset+w, c] =  \
                sprite[:,:,c] * (sprite[:,:,3]/255.0) +  frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1.0 - sprite[:,:,3]/255.0)
        return frame

    #Adjust the given sprite to the head's width and position
    #in case of the sprite not fitting the screen in the top, the sprite should be trimed
    @staticmethod
    def __adjust_sprite2head(sprite, head_width, head_ypos, ontop = True):
        (h_sprite,w_sprite) = (sprite.shape[0], sprite.shape[1])
        factor = 1.0*head_width/w_sprite
        sprite = cv2.resize(sprite, (0,0), fx=factor, fy=factor) # adjust to have the same width as head
        (h_sprite,w_sprite) = (sprite.shape[0], sprite.shape[1])

        y_orig =  head_ypos-h_sprite if ontop else head_ypos # adjust the position of sprite to end where the head begins
        if (y_orig < 0): #check if the head is not to close to the top of the image and the sprite would not fit in the screen
                sprite = sprite[abs(y_orig)::,:,:] #in that case, we cut the sprite
                y_orig = 0 #the sprite then begins at the top of the image
        return (sprite, y_orig)

    @staticmethod
    def __get_face_boundbox(points, face, face_part):
        if face_part == 1:
            (x,y,w,h) = ImageProcessor.__calculate_boundbox(points[17:22]) #left eyebrow
        elif face_part == 2:
            (x,y,w,h) = ImageProcessor.__calculate_boundbox(points[22:27]) #right eyebrow
        elif face_part == 3:
            (x,y,w,h) = ImageProcessor.__calculate_boundbox(points[36:42]) #left eye
        elif face_part == 4:
            (x,y,w,h) = ImageProcessor.__calculate_boundbox(points[42:48]) #right eye
        elif face_part == 5:
            (x,y,w,h) = ImageProcessor.__calculate_boundbox(points[29:36]) #nose
        elif face_part == 6:
            (x,y,w,h) = ImageProcessor.__calculate_boundbox(points[48:68]) #mouth
        elif face_part == 7:
            # compute the rotated box for forehead
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            h5 = h * 0.5  # scale forehead according to face bb height
            x5_1 = points[17][0]  # left eyebrow
            y5_1 = points[17][1]
            x5_2 = points[26][0]  # right eyebrow
            y5_2 = points[26][1]
            incl = ImageProcessor.__calculate_inclination(points[17], points[26])
            x5_3 = x5_2 + int(h5 * math.sin(incl * math.pi / 180.0))
            y5_3 = y5_2 - int(h5 * math.cos(incl * math.pi / 180.0))
            x5_4 = x5_1 + int(h5 * math.sin(incl * math.pi / 180.0))
            y5_4 = y5_1 - int(h5 * math.cos(incl * math.pi / 180.0))
            (x, y, w, h) = ImageProcessor.__calculate_boundbox(
                np.array([[x5_1, y5_1], [x5_2, y5_2], [x5_3, y5_3], [x5_4, y5_4]])
            )  #forehead
        elif face_part == 8:
            # compute the rotated box for top of head
            incl = ImageProcessor.__calculate_inclination(points[17], points[26])
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            h5 = h * 0.2  # height from eyebrow to bottom of the region
            h6 = h * 0.5  # height of the region
            x5_1 = points[17][0] + int(h5 * math.sin(incl * math.pi / 180.0))
            y5_1 = points[17][1] - int(h5 * math.cos(incl * math.pi / 180.0))
            x5_2 = points[26][0] + int(h5 * math.sin(incl * math.pi / 180.0))
            y5_2 = points[26][1] - int(h5 * math.cos(incl * math.pi / 180.0))
            x5_3 = x5_2 + int(h6 * math.sin(incl * math.pi / 180.0))
            y5_3 = y5_2 - int(h6 * math.cos(incl * math.pi / 180.0))
            x5_4 = x5_1 + int(h6 * math.sin(incl * math.pi / 180.0))
            y5_4 = y5_1 - int(h6 * math.cos(incl * math.pi / 180.0))
            (x, y, w, h) = ImageProcessor.__calculate_boundbox(
                np.array([[x5_1, y5_1], [x5_2, y5_2], [x5_3, y5_3], [x5_4, y5_4]])
            )  #top of head
        return (x,y,w,h)

    @staticmethod
    def __calculate_boundbox(list_coordinates):
        x = min(list_coordinates[:,0])
        y = min(list_coordinates[:,1])
        w = max(list_coordinates[:,0]) - x
        h = max(list_coordinates[:,1]) - y
        return (x,y,w,h)


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@click.group()
def main():
    pass


@main.command()
@click.argument('path', type=click.Path(exists=True))
def check_hdf5(path):
    with h5py.File(path, "r") as hf:
        data_orig = hf["train_orig"]
        data_transformed = hf["train_transformed"]
        for i in range(data_orig.shape[0]):
            plt.figure()
            # CxHxW -> HxWxC
            img = data_orig[i, :, :, :].transpose(1, 2, 0)
            img2 = data_transformed[i, :, :, :].transpose(1, 2, 0)
            img = np.concatenate((img, img2), axis=1)
            plt.imshow(img)
            plt.show()
            plt.clf()
            plt.close()


@main.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--max_count', '-n', type=click.IntRange(1, math.inf))
@click.option('--shuffle/--no_shuffle', default=True)
def create_hdf5(input_dir, output_path, max_count, shuffle):
    logger = logging.getLogger(__name__)
    logger.info('Generating HDF5')

    database_orig_path = os.path.join(input_dir, INTERIM_ORIG_DIR)
    database_transformed_path = os.path.join(input_dir, INTERIM_TRANSFORMED_DIR)
    if not os.path.exists(database_orig_path):
        raise IOError('Dir "{}" does not exist'.format(INTERIM_ORIG_DIR))
    if not os.path.exists(database_transformed_path):
        raise IOError('Dir "{}" does not exist'.format(INTERIM_TRANSFORMED_DIR))

    orig_images = [img for img in Path(database_orig_path).glob('**/*.jpg')]
    transformed_images = [img for img in Path(database_transformed_path).glob('**/*.jpg')]
    if len(orig_images) != len(transformed_images):
        raise IOError('No. of orig images ({}) does not match transformed images ({})'.format(
            len(orig_images), len(transformed_images)
        ))
    for i in range(len(orig_images)):
        if os.path.basename(str(orig_images[i])) != os.path.basename(str(transformed_images[i])):
            raise IOError('Orig images do not match transformed images')

    image_names = [os.path.basename(str(img)) for img in orig_images]

    # shuffle
    if shuffle:
        random.seed(0)
        random.shuffle(image_names)

    # limit max images
    if max_count is not None:
        image_names = image_names[:max_count]

    n = len(image_names)
    datasets = OrderedDict([
        ("train", image_names[:int(DB_SPLIT_TRAIN*n)]),
        ("val", image_names[int(DB_SPLIT_TRAIN*n):int((DB_SPLIT_TRAIN+DB_SPLIT_VALIDATION)*n)]),
        ("test", image_names[int((DB_SPLIT_TRAIN+DB_SPLIT_VALIDATION)*n):])
    ])

    def load_images(name):
        image_orig_path = os.path.join(database_orig_path, name)
        image_transformed_path = os.path.join(database_transformed_path, name)
        img_orig = cv2.imread(image_orig_path)
        img_orig = img_orig[:, :, ::-1]  # BGR to RGB
        img_transformed = cv2.imread(image_transformed_path)
        img_transformed = img_transformed[:, :, ::-1]  # BGR to RGB
        # HxWxC -> CxHxW
        img_orig = np.expand_dims(img_orig, 0).transpose([0, 3, 1, 2])
        img_transformed = np.expand_dims(img_transformed, 0).transpose([0, 3, 1, 2])
        return img_orig, img_transformed

    # determine image height and width
    sample_img, _ = load_images(image_names[0])
    nb_channels, height, width = sample_img.shape[1:]

    output_database_path = os.path.join(str(output_path))
    with h5py.File(output_database_path, "w") as hfw:
        for dataset_type, dataset in datasets.items():
            data_orig = hfw.create_dataset("%s_orig" % dataset_type,
                                           (0, nb_channels, height, width),
                                           maxshape=(None, nb_channels, height, width),
                                           dtype=np.uint8)

            data_transformed = hfw.create_dataset("%s_transformed" % dataset_type,
                                             (0, nb_channels, height, width),
                                             maxshape=(None, nb_channels, height, width),
                                             dtype=np.uint8)

            dataset_arr = np.array(dataset)
            num_files = len(dataset_arr)
            chunk_size = min(100, num_files)
            num_chunks = num_files / chunk_size
            arr_chunks = np.array_split(np.arange(num_files), num_chunks)

            with tqdm(arr_chunks, total=num_files, desc=dataset_type) as progress_bar:
                for chunk_idx in arr_chunks:
                    chunk_image_names = dataset_arr[chunk_idx].tolist()
                    output = parmap.map(load_images, chunk_image_names, pm_parallel=False)

                    arr_img_orig = np.concatenate([o[0] for o in output], axis=0)
                    arr_img_transformed = np.concatenate([o[1] for o in output], axis=0)

                    # Resize HDF5 dataset
                    data_orig.resize(data_orig.shape[0] + arr_img_orig.shape[0], axis=0)
                    data_transformed.resize(data_transformed.shape[0] + arr_img_transformed.shape[0], axis=0)

                    data_orig[-arr_img_orig.shape[0]:] = arr_img_orig.astype(np.uint8)
                    data_transformed[-arr_img_transformed.shape[0]:] = arr_img_transformed.astype(np.uint8)
                    progress_bar.update(len(chunk_idx))


@main.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--max_count', '-n', type=click.IntRange(1, math.inf))
@click.option('--output_size', type=(int, int), default=(256, 256))
@click.option('--preserve_dir/--no_preserve_dir', default=True,
              help='Preserve the original directory structure (default=True)')
@click.option('--ignore_duplicates/--no_ignore_duplicates', default=True,
              help='Ignore duplicate images (default=True)')
def apply_filter(input_dir,
                 output_dir,
                 max_count,
                 output_size,
                 preserve_dir,
                 ignore_duplicates):
    logger = logging.getLogger(__name__)
    logger.info('Applying filter to raw images')

    # create interim directory
    database_interim_path = output_dir
    if os.path.exists(database_interim_path):
        raise IOError('Output directory already exists')
    os.makedirs(database_interim_path)

    database_orig_path = None
    database_transformed_path = None
    if not preserve_dir:
        database_orig_path = os.path.join(database_interim_path, INTERIM_ORIG_DIR)
        database_transformed_path = os.path.join(database_interim_path, INTERIM_TRANSFORMED_DIR)
        os.makedirs(database_orig_path)
        os.makedirs(database_transformed_path)

    # grab raw images
    raw_images = [img for img in Path(input_dir).glob('**/*.jpg')]
    if max_count is not None:
        raw_images = raw_images[:max_count]

    # extract faces
    num_accepted = 0
    face_detector = FaceDetector(os.path.join(str(project_dir), 'src/data/caascades'))
    image_processor = ImageProcessor(output_width=output_size[0],
                                     output_height=output_size[1],
                                     sprites_path=os.path.join(str(project_dir), 'src/data/sprites'))

    # image hashes
    hashes = set()

    with tqdm(raw_images) as progress_bar:
        for raw_image_path in progress_bar:
            raw_image_path = str(raw_image_path)
            base_path = os.path.relpath(os.path.dirname(raw_image_path),
                                        start=input_dir)

            # skip if image is duplicate
            if ignore_duplicates:
                file_hash = md5(raw_image_path)
                if file_hash in hashes:
                    continue
                hashes.add(file_hash)

            # load color (BGR) image
            try:
                img = cv2.imread(raw_image_path)
                resized_image = image_processor.resize(img)
                has_face = face_detector.has_face(resized_image)
            except cv2.error:
                continue

            if has_face:
                face, landmarks = face_detector.get_landmarks(resized_image)
                doggy_image = image_processor.process(resized_image, face, landmarks, ImageProcessor.Filter.DOG)

                num_accepted += 1
                filename = os.path.basename(raw_image_path)

                # Write the images
                #    filtered -> orig
                #    face -> transformed

                if preserve_dir:
                    database_orig_path = os.path.join(database_interim_path,
                                                      base_path,
                                                      INTERIM_ORIG_DIR)
                    database_transformed_path = os.path.join(database_interim_path,
                                                             base_path,
                                                             INTERIM_TRANSFORMED_DIR)
                    if not os.path.exists(database_orig_path):
                        os.makedirs(database_orig_path)
                    if not os.path.exists(database_transformed_path):
                        os.makedirs(database_transformed_path)
                else:
                    # append base path to filename to prevent collisions
                    if base_path != '.':
                        filename = base_path.replace(os.path.sep, '_') + '_' + filename

                cv2.imwrite(os.path.join(database_orig_path, filename), doggy_image)
                cv2.imwrite(os.path.join(database_transformed_path, filename), resized_image)
                progress_bar.set_description(
                    "Wrote image file {} (accepted={})".format(filename, num_accepted)
                )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
