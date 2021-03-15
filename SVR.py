import tensorflow as tf
from model import SR_QP
import pathlib
import os
import glob
import numpy as np
import utilty as util
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

IMG_SHAPE = (None, None, 1)
layer=np.array(([192,160,160,128,128,96,96,96,64,64,32,32]))
super_reslution=SR_QP(kernel_size=3,drop_rate=0.1,layers_set=layer)
inputs = tf.keras.Input(shape=IMG_SHAPE)
predictions=super_reslution(inputs)

full_model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
#print(latest)
full_model.load_weights(latest)
input_image2 = util.load_image('test_img/baby.png', print_console=False)
full_model.summary()

#print(full_model.get_layer('conv1').weights)

out=util.apply_SR(model=full_model,input_image=input_image2,scale=2)
print(out)
print(np.max(out))
input_image=tf.expand_dims(input_image2,axis=0)
#print(tf.image.decode_jpeg(input_image2, channels=3))

plt.imshow(out)
plt.show()




