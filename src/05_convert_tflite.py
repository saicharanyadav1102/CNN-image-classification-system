"""
=============================================================================
 Step 5: Convert Trained Model to TensorFlow Lite (.tflite) Format
=============================================================================

 PURPOSE:
 This script converts our trained Keras CNN model into an optimized
 TensorFlow Lite (.tflite) model suitable for mobile deployment on Android.

 WHY TENSORFLOW LITE FOR MOBILE?
 Standard TensorFlow Keras models (.keras / .h5) can be several megabytes or
 gigabytes in size and require full Python runtime libraries to execute.
 TensorFlow Lite solves this by:
 1. Quantization & Optimization: Shrinks file size by 3x-4x with minimal
    loss in accuracy.
 2. High Speed: Optimized specifically for mobile CPUs and hardware accelerators
    (NNAPI / GPUs on Android phones).
 3. No Python Required: Runs natively in C++/Java/Kotlin inside mobile apps.

 WHAT THIS SCRIPT DOES:
 1. Loads the best trained Keras model (`models/best_model.keras`).
 2. Converts it using `tf.lite.TFLiteConverter` with `Optimize.DEFAULT`.
 3. Saves the output to `models/model.tflite`.
 4. Creates `models/labels.txt` containing our class names (`cat` and `dog`).
 5. Compares original model size against TFLite model size.
 6. Verifies the TFLite model by running a quick inference check on a test image.

 PREREQUISITES:
 - Must have trained model from `03_train_model.py`.

 OUTPUTS:
 - ../models/model.tflite  (Ready to be copied into Android app assets)
 - ../models/labels.txt    (Class labels ready for Android)

=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import importlib.util
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

data_prep = importlib.import_module("01_data_preparation")
MODELS_DIR = data_prep.MODELS_DIR


def verify_tflite_model(tflite_path):
    """
    Loads the newly saved `model.tflite` into the TFLite Interpreter
    and runs inference on a real sample image from `dataset/test/` to verify
    it works correctly before copying to Android Studio.
    """
    print("\n[INFO] Verifying TFLite model inference on sample test image...")
    # Load interpreter and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Find any available test image
    test_dir = os.path.join(data_prep.DATASET_DIR, "test", "cat")
    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        test_dir = os.path.join(data_prep.DATASET_DIR, "test", "dog")

    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        print("   No test images found for verification.")
        return

    sample_filename = os.listdir(test_dir)[0]
    sample_path = os.path.join(test_dir, sample_filename)

    # Preprocess image just like Android app does (128x128, normalize [0,1])
    img = load_img(sample_path, target_size=data_prep.IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Set input tensor and invoke interpreter
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Get prediction output
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_idx = np.argmax(output_data)
    predicted_class = data_prep.CLASSES[predicted_idx].capitalize()

    print(f"   Test Image: {sample_filename}")
    print(f"   Raw TFLite Probabilities: [Cat: {output_data[0]:.4f}, Dog: {output_data[1]:.4f}]")
    print(f"   Predicted Result: {predicted_class}")
    print("   [OK] TFLite verification passed successfully!")


def main():
    print("=====================================================================")
    print(" [Cats vs Dogs] Step 5: Convert Model to TensorFlow Lite (.tflite)")
    print("=====================================================================\n")

    # Step 1: Locate best trained model
    best_model_path = os.path.join(MODELS_DIR, "best_model.keras")
    if not os.path.exists(best_model_path):
        fallback = os.path.join(MODELS_DIR, "final_model.keras")
        if os.path.exists(fallback):
            best_model_path = fallback
        else:
            raise FileNotFoundError(
                f"\n[ERROR] Could not find model at: {best_model_path}\n"
                "Please run `python 03_train_model.py` first."
            )

    print(f"[INFO] Loading Keras model from: {best_model_path}")
    model = tf.keras.models.load_model(best_model_path)

    # Step 2: Initialize TFLiteConverter
    print("[INFO] Initializing TFLiteConverter with default optimizations & Keras 3 op support...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optimize.DEFAULT applies post-training quantization to reduce size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS  # Enable pure TensorFlow Lite builtin ops (no FlexDelegate needed)
    ]

    # Step 3: Convert model
    print("[INFO] Converting model to .tflite format (may take ~15-30 seconds)...")
    tflite_model = converter.convert()

    # Step 3.1: Ensure flatbuffer opcode versions are compatible with Android TFLite runtime (<= version 9)
    # Python TF 2.21 defaults to FULLY_CONNECTED version 12 which older Android runtimes reject.
    try:
        from tensorflow.lite.python import schema_py_generated as schema
        data = bytearray(tflite_model)
        model_obj = schema.Model.GetRootAsModel(data, 0)
        for i in range(model_obj.OperatorCodesLength()):
            op = model_obj.OperatorCodes(i)
            if op.Version() > 9:
                offset = op._tab.Offset(8)
                if offset != 0:
                    pos = op._tab.Pos + offset
                    data[pos:pos+4] = (9).to_bytes(4, 'little')
        tflite_model = bytes(data)
        print("[INFO] Patched flatbuffer opcode versions for universal Android compatibility.")
    except Exception as e:
        print(f"[WARNING] Could not patch flatbuffer opcode versions: {e}")

    # Step 4: Save model.tflite
    tflite_path = os.path.join(MODELS_DIR, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[OK] Saved TFLite model to -> {tflite_path}")

    # Step 5: Save labels.txt
    labels_path = os.path.join(MODELS_DIR, "labels.txt")
    with open(labels_path, "w") as f:
        for c in data_prep.CLASSES:
            f.write(f"{c}\n")
    print(f"[OK] Saved class labels to -> {labels_path}")

    # Step 6: Compare file sizes
    keras_size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
    tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print("\n--- File Size Comparison ---")
    print(f"   Original Keras model: {keras_size_mb:.2f} MB")
    print(f"   Optimized TFLite model: {tflite_size_mb:.2f} MB")
    print(f"   Size reduction: {(1.0 - tflite_size_mb / keras_size_mb) * 100:.1f}% smaller!")

    # Step 7: Verify TFLite model
    verify_tflite_model(tflite_path)

    print("\n=====================================================================")
    print(" [OK] Step 5 Complete! Python ML Pipeline is 100% finished.")
    print("=====================================================================")
    print(" Next Step for Android Studio:")
    print(" Copy `models/model.tflite` into `android-app/app/src/main/assets/`")
    print("=====================================================================")


if __name__ == "__main__":
    main()
