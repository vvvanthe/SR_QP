import tensorflow as tf

from tensorflow.keras.initializers import Constant
class SR_QP(tf.keras.Model):
    def __init__(self, kernel_size,layers_set,drop_rate,transpose_conv=False,BatchNorm=True,alpha=0.25):
        super().__init__()
        self.drop_rate=drop_rate
        self.kernel_size=(kernel_size,kernel_size)
        self.layers_set=layers_set
        self.transcov2=transpose_conv
        self.BatchNorm=BatchNorm
        self.apl=alpha

    def __call__(self, inputs):

        list_out=[]
        x=inputs
        list_fin=[]

        for i in range(len(self.layers_set)):
            if i ==0:
                size=(9,9)
            elif i==1 or i==2:
                size = (6, 6)
            elif i==3:
                size = (4, 4)
            else:
                size=self.kernel_size
            x=tf.keras.layers.Conv2D(filters=self.layers_set[i],use_bias=True,kernel_size=size,padding='same',name='conv'+str(i))(x)
            if self.BatchNorm:
                x=tf.keras.layers.BatchNormalization()(x)

            #x=tf.keras.activations.tanh(x)
            x=tf.keras.layers.LeakyReLU(alpha= self.apl)(x)
            x=tf.keras.layers.Dropout(self.drop_rate)(x)
            if not self.BatchNorm:
                x=tf.clip_by_value( x, -5, 5, name=None)

            list_out.append(x)

        concate=tf.concat(list_out, -1, name="H_concat")

        pre_upsr=tf.keras.layers.Conv2D(filters=4,use_bias=True,kernel_size=self.kernel_size,padding='same',name='conv'+str(i+1))(concate)
        if self.BatchNorm:
            pre_upsr = tf.keras.layers.BatchNormalization()(pre_upsr)
        #pre_upsr = tf.keras.activations.tanh(pre_upsr)
        pre_upsr = tf.keras.layers.LeakyReLU(alpha= self.apl)(pre_upsr)
        pre_upsr = tf.keras.layers.Dropout(self.drop_rate)(pre_upsr)

        output = tf.nn.depth_to_space(pre_upsr,2)
        #output = tf.keras.activations.tanh(output)
        output = tf.keras.layers.LeakyReLU(alpha= self.apl)(output)

        if not self.BatchNorm:
            output = tf.clip_by_value(output, -5, 5, name=None)

        if not self.transcov2:
            return output

        list_fin.append(output)

        trans_out=tf.keras.layers.Conv2DTranspose(filters=2,use_bias=True, kernel_size=self.kernel_size,padding='same', name='transpose2D',strides=(2,2))(concate)
        if self.BatchNorm:
            trans_out = tf.keras.layers.BatchNormalization()(trans_out)

        #trans_out = tf.keras.activations.tanh(trans_out)


        trans_out = tf.keras.layers.LeakyReLU(alpha= self.apl)(trans_out)

        if not self.BatchNorm:
            trans_out = tf.clip_by_value(trans_out, -5, 5, name=None)



        list_fin.append(trans_out)

        concate_2=tf.concat(list_fin, -1, name="H_concat_2")

        SR_x2 = tf.keras.layers.Conv2D(filters=1, use_bias=True, kernel_size=self.kernel_size, padding='same',
                                   name='conv' + str(i+2))(concate_2)

        if self.BatchNorm:
            SR_x2 = tf.keras.layers.BatchNormalization()(SR_x2)

        #SR_x2 = tf.keras.activations.tanh(SR_x2)
        SR_x2 = tf.keras.layers.LeakyReLU(alpha= self.apl)(SR_x2)
        SR_x2 = tf.keras.layers.Dropout(self.drop_rate)(SR_x2)

        if not self.BatchNorm:
            SR_x2 = tf.clip_by_value(SR_x2, -5, 5, name=None)

        return SR_x2



