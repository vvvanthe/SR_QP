import tensorflow as tf
from model import SR_QP
import pathlib
import os
import glob
import numpy as np
import utilty as util
import matplotlib.pyplot as plt

Root_path_in='Input/'
in_image='QP51'

layer= np.array([32, 32, 32, 16, 16, 8, 8])#np.array([32, 32, 16, 16, 16, 8, 8])#np.array(([192,160,160,128,128,96,96,96,64,64,32,32]))

super_reslution=SR_QP(kernel_size=3,drop_rate=0.1,layers_set=layer,transpose_conv=True)

IMG_SHAPE = (None, None, 1)

inputs = tf.keras.Input(shape=IMG_SHAPE)
predictions=super_reslution(inputs)

full_model = tf.keras.models.Model(inputs=inputs, outputs=predictions)


checkpoint_path = "checkpoints/20210310/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest="checkpoints/20210312/cp-0030.ckpt"
print(latest)
full_model.load_weights(latest)



for i, image_path in enumerate(sorted(glob.glob(Root_path_in+ in_image+ "/*"))):
   image = util.load_image(image_path, print_console=False)
   name = os.path.basename(image_path)
   filename, extension = os.path.splitext(name)
   out = util.apply_SR(model=full_model, input_image=image, scale=2)
   util.save_image('Output/'+in_image+'_30/'+filename+'.png',out)
