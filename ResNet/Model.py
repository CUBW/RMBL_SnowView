import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model



def resnet_model(img_height, img_width, img_channels):
    # Load ResNet50v2 without including the top layers
    base_model = ResNet50V2(input_shape=(img_height, img_width, img_channels), include_top=False)

    # Define encoder (downsampling path)
    encoder = Model(inputs=base_model.input, outputs=base_model.get_layer('conv4_block6_out').output)

# Debug: Print encoder output shape
    print(f"Encoder output shape: {encoder.output_shape}")

    # Define decoder (upsampling path)
    input_encoder = Input(shape=encoder.output_shape[1:])
    up1 = UpSampling2D((2, 2))(input_encoder)
# Debug: Print shapes before first concatenate
    print(f"Shape before first concatenate: {encoder.output_shape}, {up1.shape}")
    # Ensure up1 matches the shape of encoder output after upsampling
    assert encoder.output_shape[1:] == up1.shape[1:], "Shapes of encoder output and up1 do not match."


    concat1 = Concatenate(axis=-1)([base_model.get_layer('conv4_block6_out').output, up1])
    conv1 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat1)

    up2 = UpSampling2D((2, 2))(conv1)
    concat2 = Concatenate(axis=-1)([base_model.get_layer('conv3_block4_out').output, up2])
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat2)

    up3 = UpSampling2D((2, 2))(conv2)
    concat3 = Concatenate(axis=-1)([base_model.get_layer('conv2_block3_out').output, up3])
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat3)

    up4 = UpSampling2D((2, 2))(conv3)
    concat4 = Concatenate(axis=-1)([base_model.get_layer('conv1_conv').output, up4])
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat4)

    # Final segmentation output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv4)

    # Create the model
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

if __name__=='__main__':
    model = resnet_model(640, 640, 3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Trainable params: {np.sum([np.prod(v._shape) for v in model.trainable_variables])}")
    print(f"Model Summary: {model.summary()}")