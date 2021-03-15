import tensorflow as tf
from model import SR_QP
import pathlib
import os
import glob
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import time
import utilty as util
import cv2

BATCH_SIZE = 15
BUFFER_SIZE=5000

data_dir = './batch_data/Depth_hevc_set6/'
data_dir = pathlib.Path(data_dir)

in_dir=os.path.join(data_dir, 'input')
#in_dir=pathlib.Path(in_dir)

bic_dir=os.path.join(data_dir, 'interpolated')
#bic_dir=pathlib.Path(bic_dir)

true_dir=os.path.join(data_dir, 'true')
#true_dir=pathlib.Path(true_dir)





image_in=sorted(glob.glob(in_dir+"/*.bmp"))#tf.data.Dataset.list_files(str(in_dir / '*'), shuffle=False)
image_true=sorted(glob.glob(true_dir+"/*.bmp"))#tf.data.Dataset.list_files(str(true_dir / '*'), shuffle=False)
image_bic=sorted(glob.glob(bic_dir+"/*.bmp"))#tf.data.Dataset.list_files(str(bic_dir / '*'), shuffle=False)

image_count = len(image_bic)



def map_func(img_in, img_true, img_bic):
    img_tensor_in=tf.io.read_file(img_in)
    img_tensor_in=tf.image.decode_bmp(img_tensor_in,channels=1)
    img_tensor_in=tf.image.convert_image_dtype(img_tensor_in, tf.float32)

    img_tensor_true = tf.io.read_file(img_true)
    img_tensor_true = tf.image.decode_bmp(img_tensor_true, channels=1)
    img_tensor_true = tf.image.convert_image_dtype(img_tensor_true, tf.float32)

    img_tensor_bic = tf.io.read_file(img_bic)
    img_tensor_bic = tf.image.decode_bmp(img_tensor_bic, channels=1)
    img_tensor_bic = tf.image.convert_image_dtype(img_tensor_bic, tf.float32)

    return img_tensor_in, img_tensor_true,img_tensor_bic


options = tf.data.Options()
options.experimental_optimization.map_fusion = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.noop_elimination = True
options.experimental_optimization.apply_default_optimizations = True

dataset = tf.data.Dataset.from_tensor_slices((image_in, image_true,image_bic))
dataset=dataset.shuffle(BUFFER_SIZE)
val_size = 0#int(image_count * 0.2/BATCH_SIZE)*BATCH_SIZE
train_size=image_count-val_size
#val_ds = dataset.take(val_size)
train_ds=dataset#.skip(val_size)

#val_ds = val_ds.map(lambda item1, item2, item3: tf.numpy_function(
#          map_func, [item1, item2, item3], [tf.float32, tf.float32, tf.float32]),
#          num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.map(lambda item1, item2, item3: tf.numpy_function(
          map_func, [item1, item2, item3], [tf.float32, tf.float32, tf.float32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
# Shuffle and batch


#val_ds = val_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#val_ds = val_ds.prefetch(1)
#val_ds = val_ds.with_options(options)

train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#train_ds = train_ds.prefetch(1)
train_ds = train_ds.with_options(options)

for (batch, (img_in, img_true,img_bic)) in enumerate(train_ds):
    print(img_true.shape)
    break


### BUILD MODEL
layer= np.array([48, 32, 32, 16, 16, 8, 8])#np.array([32, 32, 16, 16, 16, 8, 8])#np.array(([192,160,160,128,128,96,96,96,64,64,32,32]))

super_reslution=SR_QP(kernel_size=3,drop_rate=0.2,layers_set=layer,transpose_conv=True)



vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
vgg16.trainable = False
loss_model = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block4_conv2').output)
for l in loss_model.layers:
    l.trainable = False

@tf.function
def vgg_loss_VGG16(y_true, y_pred):
    y_pred3=tf.keras.layers.concatenate([y_pred, y_pred, y_pred])*255.
    y_true3=tf.keras.layers.concatenate([y_true, y_true, y_true])*255.
    return K.mean(K.square(loss_model(y_true3) - loss_model(y_pred3)))



@tf.function
def loss_mse(y_true, y_pred):
    diff = tf.subtract(y_true*255., y_pred*255.)
    mse = tf.reduce_mean(tf.square(diff))
    return mse



for (batch, (img_in, img_true,img_bic)) in enumerate(train_ds):
    diff = tf.subtract(img_true, img_bic, "diff")
    mse = loss_mse(img_true,img_bic)
    print(mse.numpy())
    vgg=vgg_loss_VGG16(img_true,img_bic)
    print(vgg.numpy())
    break


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=20000,
    decay_rate=0.97)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,clipnorm=1,clipvalue=5000) #clipnorm=1,clipvalue=5000




'''
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(super_reslution)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)
'''
start_epoch = 0
EPOCHS = 80

IMG_SHAPE = (None, None, 1)

inputs = tf.keras.Input(shape=IMG_SHAPE)
predictions=super_reslution(inputs)

full_model = tf.keras.models.Model(inputs=inputs, outputs=predictions)


@tf.function
def train_step(input_img, target_img,bic_img):

    with tf.GradientTape() as tape:
        out_sr=full_model(input_img)

        mse=loss_mse(target_img,tf.add(out_sr,bic_img))
        loss_img=loss_mse(target_img,tf.add(out_sr,bic_img))


    trainable_variables = full_model.trainable_variables
    gradients = tape.gradient(loss_img, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return mse,loss_img,out_sr

checkpoint_path = "checkpoints/20210315/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)






for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_img_loss = 0
    total_mse_loss = 0
    total_mse_valid= 0
    total_img_valid=0
    PSNR = []
    for (batch, (img_in, img_true,img_bic)) in enumerate(train_ds):
        mse_loss, img_loss,out_sr = train_step(img_in, img_true,img_bic)
        total_img_loss += img_loss
        total_mse_loss +=mse_loss


        if batch % 5000 == 0 and batch>0:
            print('Epoch {} Batch {} MSE {} VGG {}'.format(
                epoch + 1, batch,total_mse_loss/(batch+1), total_img_loss/(batch+1)))

            print(np.max(out_sr.numpy()))
            print(np.min(out_sr.numpy()))

    for i, image_path in enumerate(sorted(glob.glob("Input/QP51/*.png"))):
        image = util.load_image(image_path, print_console=False)
        out = util.apply_SR(model=full_model, input_image=image, scale=2)

        name = os.path.basename(image_path)
        file_root = util.load_image('Input/or/' + name, print_console=False)
        ps, _ = util.compute_psnr_and_ssim(out, file_root)
        print("image {}: {}".format(i + 1, ps))
        PSNR.append(ps)


    print('Epoch {} VALID PSNR {}'.format(
        epoch + 1, np.mean(PSNR)))


    if epoch % 1 == 0:
        full_model.save_weights(checkpoint_path.format(epoch=(epoch+1)))
    #print(full_model.get_layer('conv1').weights)
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))






