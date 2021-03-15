import numpy as np
import glob
import utilty as util
import os
from PIL import Image
import math
batch_image_size=64
scale=2



INPUT_IMAGE_DIR='input'
INTERPOLATED_IMAGE_DIR='interpolated'
TRUE_IMAGE_DIR='true'

dataset='Depth_hevc_set6'
full_data='Data'+"/"+dataset
batch_dir='batch_data'+"/"+dataset

in_image_dir=full_data+ "/" + INPUT_IMAGE_DIR+ "/"
tr_image_dir=full_data+ "/" + TRUE_IMAGE_DIR+ "/"


util.make_dir(batch_dir)
util.clean_dir(batch_dir)
util.make_dir(batch_dir + "/" + INPUT_IMAGE_DIR)
util.make_dir(batch_dir + "/" + INTERPOLATED_IMAGE_DIR)
util.make_dir(batch_dir + "/" + TRUE_IMAGE_DIR)

output_window_size = batch_image_size * scale
stride=batch_image_size//2
output_window_stride = stride * scale

def build_image_set(file_truepath,scale,convert_ycbcr=False,resampling_method="bicubic",print_console=True):
    true_image = util.set_image_alignment(util.load_image(file_truepath, print_console=print_console), scale)
    if true_image.shape[2] == 3:
        name = os.path.basename(file_truepath)
        filename, extension = os.path.splitext(name)
        file_inpath=in_image_dir+ "/" +name
        #input_image =util.resize_image_by_pil(true_image, 1./scale, resampling_method=resampling_method)
        input_image=util.load_image(file_inpath, print_console=False)
        input_interpolated_image = util.resize_image_by_pil(input_image, scale, resampling_method=resampling_method)

        if convert_ycbcr:
            input_image=util.convert_rgb_to_y(input_image)
            input_interpolated_image=util.convert_rgb_to_y(input_interpolated_image)
            true_image=util.convert_rgb_to_y(true_image)

        return input_image, input_interpolated_image, true_image


def save_batch_image( image_number, image,type):
    return util.save_image(batch_dir + "/" + type + "/%06d.bmp" % image_number, image)

images_count = 0

for i, orginal_image in enumerate(sorted(glob.glob(tr_image_dir + "*"))):
    input_image, input_interpolated_image, true_image =  build_image_set(orginal_image,scale=scale,convert_ycbcr=True)
    input_batch_images = util.get_split_with_channels(input_image, batch_image_size, stride=stride)

    input_interpolated_batch_images = util.get_split_with_channels(input_interpolated_image, output_window_size,
                                                            stride=output_window_stride)

    if input_batch_images is None or input_interpolated_batch_images is None:
        # if the original image size * scale is less than batch image size
        continue
    input_count = input_batch_images.shape[0]
    true_batch_images = util.get_split_with_channels(true_image, output_window_size, stride=output_window_stride)

    for i in range(input_count):
        save_batch_image(images_count, input_batch_images[i],INPUT_IMAGE_DIR)
        save_batch_image(images_count, input_interpolated_batch_images[i], INTERPOLATED_IMAGE_DIR)
        save_batch_image(images_count, true_batch_images[i], TRUE_IMAGE_DIR)
        images_count += 1

