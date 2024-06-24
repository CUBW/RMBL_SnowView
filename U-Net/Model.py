
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


def decoder_block(conv, pool, n_filters, dropout, l2):
    """
    Function to create a decoder block with a Conv2DTranspose layer followed by a concatenation of the skip connections and a Conv2D layer.
    
    Args:
    - conv: Tensor from the encoder block.
    - pool: Tensor from the encoder block.
    - n_filters: Number of filters for the Conv2D layer.
    - l2: L2 regularization strength.
    
    Returns:
    - conv: Output tensor after applying the Conv2D layer.
    """
    
    # Conv2DTranspose Layer
    conv = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', kernel_regularizer= tf.keras.regularizers.L2(l2))(conv)
    # Concatenate the skip connections
    concat = concatenate([conv, pool])
    
    # Convolutional Block
    conv = conv_block(concat, n_filters, dropout, l2)
    return conv

    

def unet_model(n_classes, img_height, img_width, img_channels):
    
    inputs = Input((img_height, img_width, img_channels))
    
    # Encoder
    conv1, pool1 = encoder_block(inputs, n_filters=16, dropout=0, l2=0.0001) 
    conv2, pool2 = encoder_block(pool1, n_filters=32, dropout=0, l2=0.0001) 
    conv3, pool3 = encoder_block(pool2, n_filters=64, dropout=0, l2=0.0001)
    conv4, pool4 = encoder_block(pool3, n_filters=128, dropout=0, l2=0.001)
    conv5, pool5 = encoder_block(pool4, n_filters=256, dropout=0, l2=0.001)
    conv6, pool6 = encoder_block(pool5, n_filters=512, dropout=0, l2=0.01)
    
    # Bottleneck
    bridge = conv_block(pool6, n_filters=1024, dropout=0, l2=0.01)
    
    # Decoder
    upsampling6 = decoder_block(bridge, conv6, n_filters=512, dropout=0, l2=0.01)
    upsampling5 = decoder_block(upsampling6, conv5, n_filters=256, dropout=0, l2=0.001)
    upsampling4 = decoder_block(upsampling5, conv4, n_filters=128, dropout=0, l2=0.001)
    upsampling3 = decoder_block(upsampling4, conv3, n_filters=64, dropout=0, l2=0.0001)
    upsampling2 = decoder_block(upsampling3, conv2, n_filters=32, dropout=0, l2=0.0001)
    upsampling1 = decoder_block(upsampling2, conv1, n_filters=16, dropout=0, l2=0.0001)

    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(upsampling1)
    
    # Create the model
    model = Model(inputs=[inputs], outputs=[outputs])
    
    
    
    
    return model

if __name__ == "__main__":
    # Create the U-Net model for binary segmentation
    model = unet_model(n_classes=1, img_height=128, img_width=128, img_channels=3)
    # Compile the model with binary crossentropy loss and Adam optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam')

