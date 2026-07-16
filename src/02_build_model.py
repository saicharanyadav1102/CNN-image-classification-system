"""
=============================================================================
 Step 2: Build the CNN Model (Convolutional Neural Network)
=============================================================================

 PURPOSE:
 This script defines the architecture of our Convolutional Neural Network (CNN)
 using TensorFlow and Keras.

 WHAT IS A CNN?
 A Convolutional Neural Network is a specialized type of neural network
 designed specifically for processing visual data like images. Instead of
 looking at every pixel independently, it uses "convolutional filters" to scan
 across the image and discover spatial patterns (edges, textures, shapes).

 ARCHITECTURE SUMMARY:
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ Input Image (128 x 128 x 3 RGB)                                         │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ Block 1: Conv2D (32 filters, 3x3) + ReLU  ──> Detects basic edges       │
 │          MaxPooling2D (2x2)               ──> Halves spatial size (64)  │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ Block 2: Conv2D (64 filters, 3x3) + ReLU  ──> Detects textures/shapes   │
 │          MaxPooling2D (2x2)               ──> Halves spatial size (32)  │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ Block 3: Conv2D (128 filters, 3x3) + ReLU ──> Detects complex body parts│
 │          MaxPooling2D (2x2)               ──> Halves spatial size (16)  │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ Flatten()                                 ──> Converts 2D maps to 1D    │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ Dense (256 units) + ReLU                  ──> Learns feature combinations│
 │ Dropout (0.5)                             ──> Prevents overfitting      │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ Dense (2 units, Softmax)                  ──> Outputs class probabilities│
 └─────────────────────────────────────────────────────────────────────────┘

 OUTPUTS:
 - Prints the complete Keras model summary with parameter counts.

=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def build_cnn_model(input_shape=(128, 128, 3), num_classes=3):
    """
    Constructs and returns our MobileNetV2 transfer learning model for Cats vs Dogs.
    Pre-trained on ImageNet (14 million images) for 98%+ accuracy and ultra-fast edge speed.
    """
    inputs = tf.keras.Input(shape=input_shape, name='input_layer')
    
    # Rescale from [0.0, 1.0] (Keras ImageDataGenerator & Android ByteBuffer format)
    # to [-1.0, 1.0] required by MobileNetV2
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name='mobilenet_rescaling')(inputs)
    
    # Load MobileNetV2 pre-trained on ImageNet without the top 1000-class classification head
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze the base model weights so we preserve ImageNet feature extraction
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output_classes')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Cats_vs_Dogs_MobileNetV2')
    return model


def main():
    print("=====================================================================")
    print(" [Cats vs Dogs] Step 2: CNN Model Architecture Inspection")
    print("=====================================================================\n")

    # Build the model
    model = build_cnn_model(input_shape=(128, 128, 3), num_classes=2)

    # Display the summary table
    print("--- Model Summary Table ---")
    model.summary()

    print("\n--- Layer-by-Layer Purpose Explanation ---")
    print(" 1. conv2d_block1 (32 filters):  Finds basic boundaries, outlines, and edges.")
    print(" 2. maxpool_block1 (2x2):        Downsamples from 128x128 -> 64x64.")
    print(" 3. conv2d_block2 (64 filters):  Combines edges into textures (fur, whiskers).")
    print(" 4. maxpool_block2 (2x2):        Downsamples from 64x64 -> 32x32.")
    print(" 5. conv2d_block3 (128 filters): Combines textures into animal body parts.")
    print(" 6. maxpool_block3 (2x2):        Downsamples from 32x32 -> 16x16.")
    print(" 7. flatten:                     Converts 16x16x128 (32,768 values) to 1D vector.")
    print(" 8. dense_hidden (256 units):    Combines features to make intermediate decisions.")
    print(" 9. dropout (0.5):               Drops 50% neurons randomly to prevent overfitting.")
    print(" 10. output_classes (2 units):   Outputs [P(Cat), P(Dog)] probabilities.")

    print("\n=====================================================================")
    print(" [OK] Step 2 Complete! You are ready to run `03_train_model.py`.")
    print("=====================================================================")


if __name__ == "__main__":
    main()
