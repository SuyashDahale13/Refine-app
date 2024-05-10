# U-Net
# Unet in function form
import tensorflow as tf 
import keras

def convb(input_tensor, n_filters, kernel_size=3):
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
    return x 

def encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3):
    f = convb(inputs, n_filters=n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(f)
    p = tf.keras.layers.Dropout(0.3)(p)

    return f, p 

def encoder(inputs):
    f1, p1 = encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3)
    f2, p2 = encoder_block(p1, n_filters=128, pool_size=(2,2), dropout=0.3)
    f3, p3 = encoder_block(p2, n_filters=256, pool_size=(2,2), dropout=0.3)
    f4, p4 = encoder_block(p3, n_filters=512, pool_size=(2,2), dropout=0.3)

    return p4, (f1, f2, f3, f4)

def bottelneck(inputs):
    bottel_neck = convb(inputs, n_filters=1024)

    return bottel_neck

def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides=strides, padding='same')(inputs)
    c = tf.keras.layers.concatenate([u,conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = convb(c, n_filters, kernel_size=3)

    return c 

def decoder(inputs, convs, output_channels):
    f1, f2, f3, f4 = convs 
    c6 = decoder_block(inputs, f4, n_filters=512, kernel_size=(3,3), strides=(2,2), dropout=0.3)
    c7 = decoder_block(c6, f3, n_filters=256, kernel_size=(3,3), strides=(2,2), dropout=0.3)
    c8 = decoder_block(c7, f2, n_filters=128, kernel_size=(3,3), strides=(2,2), dropout=0.3)
    c9 = decoder_block(c8, f1, n_filters=64, kernel_size=(3,3), strides=(2,2), dropout=0.3)

    outputs = tf.keras.layers.Conv2D(output_channels, (1,1), activation='softmax')(c9)

    return outputs

#######################################################################################################################################
OUTPUT_CHANNELS = 4
#######################################################################################################################################

def unet(inputs):

    # Feed the inputs to the encoder
    encoder_output , convs = encoder(inputs)

    # Feed the encoder output to the bottelneck
    bottel_neck = bottelneck(encoder_output)

    # Feed the bottelneck and encoderblock outputs to the decoder
    # Specify the number of classes via the 'OUTPUT_CHANNELS' argument
    outputs = decoder(bottel_neck, convs, output_channels=OUTPUT_CHANNELS)

    return outputs

#######################################################################################################################################

# Fine Feature Path 
def block(inputs, n_filters, kernel_size=(3,3)):
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer='he_normal', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer='he_normal', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.concatenate([inputs,x])
    return x 

#####################################################################################################################################
def finefeaturepath(inputs):
    l1 = block(inputs=inputs, n_filters=64, kernel_size=(3,3))
    l2 = block(inputs=l1, n_filters=64, kernel_size=(3,3))
    l3 = block(inputs=l2, n_filters=64, kernel_size=(3,3))
    l4 = block(inputs=l3, n_filters=64, kernel_size=(3,3))

    outputs = tf.keras.layers.Conv2D(filters=OUTPUT_CHANNELS, kernel_size=(1,1))(l4)

    return outputs

#####################################################################################################################################