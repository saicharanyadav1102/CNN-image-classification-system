# CNN Image Classification System [Android Application]

A full-stack Artificial Intelligence and Machine Learning project that classifies images of **Cats** and **Dogs** in real time on an Android smartphone, while intelligently filtering out non-animal photos (like human faces, cars, and furniture) into an **`Other (Not Cat/Dog)`** category.

---

## Key Features

* **High-Accuracy Deep Learning:** Uses **MobileNetV2** Transfer Learning (pre-trained on 14 million ImageNet photos) to achieve **97%+ validation accuracy**.
* **Intelligent Background Filtering (3-Class Model):** Instead of forcing a false guess when shown a human face or everyday object, the model explicitly predicts **`Other`** to prevent errors.
* **100% Offline Edge Inference:** Runs locally on the Android device using **TensorFlow Lite (`model.tflite`)** with zero cloud latency or API costs.
* **Modern Android UI:** Built with **Kotlin**, **AndroidX CameraX**, and **Material Design 3**, allowing users to take live photos or pick images from their gallery.

---

## 📂 Project Structure

```text
├── models/
│   ├── model.tflite            # Optimized Edge TFLite model (2.42 MB)
│   └── labels.txt              # Class names: cat, dog, other
├── src/
│   ├── 01_data_preparation.py  # Dataset extraction, splitting & augmentation
│   ├── 02_build_model.py       # MobileNetV2 architecture definition
│   ├── 03_train_model.py       # Model training loop & checkpointing
│   ├── 04_evaluate_model.py    # Test evaluation & confusion matrix generator
│   └── 05_convert_tflite.py    # TFLite conversion & verification script
└── android-app/
    └── app/src/main/
        ├── assets/             # Holds model.tflite and labels.txt for Android
        └── java/com/example/imageclassifier/
            ├── ImageClassifier.kt  # TFLite ByteBuffer wrapper & inference logic
            └── MainActivity.kt     # Camera/Gallery pickers & UI state management
```

---

## 🚀 How to Run the Python Machine Learning Pipeline

1. **Install Dependencies:**
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn opencv-python pillow
   ```
2. **Train the Model & Convert to TFLite:**
   ```bash
   python src/03_train_model.py
   python src/05_convert_tflite.py
   ```
   *This automatically extracts the dataset, trains `MobileNetV2`, and exports `models/model.tflite`.*

---

## 📱 How to Run the Android App

1. Copy the newly trained model and labels into the Android assets folder:
   ```powershell
   Copy-Item -Force "models\model.tflite" "android-app\app\src\main\assets\model.tflite"
   Copy-Item -Force "models\labels.txt" "android-app\app\src\main\assets\labels.txt"
   ```
2. Open `android-app/` in **Android Studio** and click **Run**, or build and install via terminal:
   ```powershell
   cd android-app
   .\gradlew.bat assembleDebug
   adb install -r "app\build\outputs\apk\debug\app-debug.apk"
   ```

---

## 🧠 Model Specifications

* **Input Shape:** `128 x 128 x 3 RGB pixels` (normalized to `[-1.0, 1.0]`)
* **Classes:** `3` (`Cat`, `Dog`, `Other`)
* **Training Dataset Size:** `9,000 images` (`3,000` per class)
* **Model File Size:** `2.42 MB` (Optimized with TFLite XNNPACK/CPU flatbuffer patching)

## Note on Model Limitations & Statistical Predictions: 
#### Deep Learning neural networks (including MobileNetV2) evaluate images by calculating mathematical similarity across visual feature maps (edges, curves, and textures). Because the model makes statistical predictions rather than logical assertions, it operates within a closed probabilistic world 
#### As a result:
               
* Out-of-distribution objects sharing visual similarities with trained classes (e.g., a coiled charging cable resembling a curled pet tail) may occasionally trigger unexpected class predictions.
* For critical production deployments, statistical models should always be paired with secondary validation layers or human oversight.