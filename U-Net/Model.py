import numpy as np



from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf


def conv_block(x, n_filters, dropout, l2):
    """
    Function to create a convolutional block with two Conv2D layers.
    
    Args:
    - x: Input tensor.
    - n_filters: Number of filters for the Conv2D layers.
    - dropout: Dropout rate for regularization.
    - l2: L2 regularization strength.
    
    Returns:
    - Output tensor after passing through the Conv2D layers.
    """
    
    # First Convolutional Layer
    conv1 = Conv2D(n_filters, (3, 3), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(l2))(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(dropout)(conv1)
    
    
    # Second Convolutional Layer
    conv2 = Conv2D(n_filters, kernel_size=(3, 3), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(l2))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    return conv2

    
    
def encoder_block(inputs , n_filters, dropout, l2):
    """
    Function to create an encoder block with a Conv2D layer followed by a MaxPooling2D layer.
    
    Args:
    - inputs: Input tensor.
    - n_filters: Number of filters for the Conv2D layer.
    - dropout: Dropout rate for regularization.
    - l2: L2 regularization strength.
      
    Returns:
    - pool: Output tensor after applying MaxPooling2D.
    - conv: Output tensor after applying the Conv2D layer.
    """
    
    # Convolutional Block
    conv = conv_block(inputs, n_filters, dropout, l2)
    
    # MaxPooling Layer
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool


# def decoder_block(pool, skip_connection, n_filters, dropout, l2):
#     """
#     Function to create a decoder block with a Conv2DTranspose layer followed by a concatenation of the skip connections and a Conv2D layer.
    
#     Args:
#     - conv: Tensor from the encoder block.
#     - pool: Tensor from the encoder block.
#     - n_filters: Number of filters for the Conv2D layer.
#     - l2: L2 regularization strength.
    
#     Returns:
#     - conv: Output tensor after applying the Conv2D layer.
#     """
    
#     # Conv2DTranspose Layer
#     upsample = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', kernel_regularizer= tf.keras.regularizers.L2(l2))(pool)
#     # Concatenate the skip connections
#     concat = concatenate([upsample, skip_connection])
    
#     # Convolutional Block
#     conv = conv_block(concat, n_filters, dropout, l2)
#     return conv

def decoder_block(pool, skip_connection, n_filters, dropout, l2):
    # Upsample using nearest neighbor and bilinear interpolation
    upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')(pool)
    upsample = Conv2D(n_filters, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2))(upsample)
    # Concatenate the skip connections
    concat = concatenate([upsample, skip_connection])
    # Convolutional Block
    conv = conv_block(concat, n_filters, dropout, l2)
    return conv




def unet_model(n_classes, img_height, img_width, img_channels):
    
    inputs = Input((img_height, img_width, img_channels))
    
    # Encoder
    conv1, pool1 = encoder_block(inputs, 16, 0, 0.0001)
    conv2, pool2 = encoder_block(pool1, 32, 0, 0.0001)
    conv3, pool3 = encoder_block(pool2, 64, 0, 0.0001)
    conv4, pool4 = encoder_block(pool3, 128, 0, 0.001)
    conv5, pool5 = encoder_block(pool4, 256, 0, 0.001)
    conv6, pool6 = encoder_block(pool5, 512, 0, 0.01)
    
    # Bottleneck
    bridge = conv_block(pool6, n_filters=1024, dropout=0, l2=0.01)
    
    # Decoder
    upsampling6 = decoder_block(bridge, conv6, 512, 0, 0.01)  # From bottleneck to first decoder layer
    upsampling5 = decoder_block(upsampling6, conv5, 256, 0, 0.001)  # From first decoder layer to second
    upsampling4 = decoder_block(upsampling5, conv4, 128, 0, 0.001)  # From second decoder layer to third
    upsampling3 = decoder_block(upsampling4, conv3, 64, 0, 0.0001)  # From third decoder layer to fourth
    upsampling2 = decoder_block(upsampling3, conv2, 32, 0, 0.0001)  # From fourth decoder layer to fifth
    upsampling1 = decoder_block(upsampling2, conv1, 16, 0, 0.0001)  # From fifth decoder layer to final decoder layer


    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(upsampling1)
    
    # Create the model
    model = Model(inputs=[inputs], outputs=[outputs])
    return model





if __name__ == "__main__":
    # Create the U-Net model for binary segmentation
    model = unet_model(n_classes=1, img_height=640, img_width=640, img_channels=3)
    # Compile the model with binary crossentropy loss and Adam optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam')
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Trainable params: {np.sum([np.prod(v._shape) for v in model.trainable_variables])}")

