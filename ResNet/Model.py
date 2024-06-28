import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2

def resnet_model(img_height, img_width, img_channels):
    # Encoder: Pre-trained ResNet50V2
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, img_channels))
    
    # Freeze the base model
    base_model.trainable = False
    
    # Encoder
    encoder_output = base_model.output

    # Decoder
    x = layers.Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(encoder_output)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Final layer
    output = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    # Create the model
    model = models.Model(inputs=base_model.input, outputs=output)
    
    return model

if __name__=='__main__':
    model = resnet_model(640, 640, 3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Trainable params: {np.sum([np.prod(v._shape) for v in model.trainable_variables])}")
    # print(f"Model Summary: {model.summary()}")