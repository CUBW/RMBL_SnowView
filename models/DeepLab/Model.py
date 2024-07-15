# Adapted from: https://github.com/keras-team/keras-io/blob/master/examples/vision/deeplabv3_plus.py
# Original author: Soumik Rakshit
# License: Apache License 2.0

# modified code for model to work for our specific use case

import keras
import tensorflow as tf
from keras import layers

def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False):
    """
    Applies a convolutional block to the input tensor.

    Args:
        block_input (tensor): The input tensor to the convolutional block.
        num_filters (int): The number of filters in the convolutional layer. Default is 256.
        kernel_size (int): The size of the convolutional kernel. Default is 3.
        dilation_rate (int): The dilation rate for the convolutional layer. Default is 1.
        use_bias (bool): Whether to use bias in the convolutional layer. Default is False.

    Returns:
        tensor: The output tensor after applying the convolutional block.
    """
    x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", use_bias=use_bias,kernel_initializer=keras.initializers.HeNormal())(block_input)
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x)

def DilatedSpatialPyramidPooling(dspp_input):
    """
    Performs Dilated Spatial Pyramid Pooling on the input tensor.

    Args:
        dspp_input (tensor): Input tensor to the Dilated Spatial Pyramid Pooling layer.

    Returns:
        tensor: Output tensor after applying Dilated Spatial Pyramid Pooling.
    """
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(size=(dims[-3]//x.shape[1], dims[-2]//x.shape[2]),interpolation="bilinear")(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeepLabV3Plus(n_classes, img_height, img_width, img_channels):
    """
    Creates a keras model based on the DeepLabV3+ architecture for semantic segmentation using resnet50 as a backbone
    
    Args:
        n_classes (int): The number of classes for semantic segmentation.
        img_height (int): The height of the input images.
        img_width (int): The width of the input images.
        img_channels (int): The number of channels in the input images.
    
    Returns:
        keras.Model: The DeepLabV3Plus model.
    """
    model_input = keras.Input(shape=(img_height, img_width, img_channels))
    preprocessed = tf.keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=preprocessed)
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(size=(img_height // 4 // x.shape[1], img_width // 4 // x.shape[2]), interpolation="bilinear")(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(size=(img_height // x.shape[1], img_width // x.shape[2]), interpolation="bilinear")(x)
    model_output = layers.Conv2D(n_classes, kernel_size=(1,1), padding="same")(x)

    return keras.Model(inputs=model_input, outputs=model_output)

if __name__ == "__main__":
    model = DeepLabV3Plus(n_classes=1, img_height=640, img_width=640, img_channels=4)
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Trainable params: {model.count_params()}")
    # print(f"Model Summary: {model.summary()}")
    