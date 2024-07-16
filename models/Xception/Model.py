import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception

def xception_model(img_height, img_width, img_channels):
    """
    Creates an Xception-based model for image segmentation using an encoder/decoder structure.

    Args:
        img_height (int): The height of the input images.
        img_width (int): The width of the input images.
        img_channels (int): The number of channels in the input images.

    Returns:
        tf.keras.Model: The Xception-based model for image segmentation.
    """
    # Load the pretrained Xception model without the top layers
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, img_channels))
    
    # Freeze the base model
    base_model.trainable = False
    
    # Encoder
    inputs = layers.Input(shape=(img_height, img_width, img_channels))
    x = base_model(inputs, training=False)
    
    # Decoder
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)
    
    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    # Create the model
    model = models.Model(inputs, outputs)
    
    return model

if __name__ == "__main__":
    model = xception_model(640, 640, 3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Trainable params: {np.sum([np.prod(v._shape) for v in model.trainable_variables])}")
    # print(f"Model Summary: {model.summary()}")
