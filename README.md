# 🐱 vs 🐶 CNN-Based Image Classification System (Cats vs Dogs) using TensorFlow Lite

A complete beginner-to-intermediate level machine learning project that builds an image classification system using a Convolutional Neural Network (CNN). The model is trained using TensorFlow on the provided `archive.zip` dataset (`Cats vs Dogs`), converted to TensorFlow Lite (`.tflite`), and integrated into a custom Android application (`Kotlin` + `CameraX`) for real-time mobile classification.

---

## 📋 Project Overview

This project implements an end-to-end AI/ML and Mobile development pipeline:

1. **Data Preparation & Augmentation** — Automatically extracts and organizes `archive.zip` (`cat` and `dog` images) into train/validation/test splits, rescales pixel values to `[0, 1]`, and applies real-time data augmentation.
2. **Model Architecture** — Builds a custom 3-block Convolutional Neural Network (`128x128` RGB input) with `Conv2D`, `MaxPooling2D`, `Dense`, and `Dropout` layers.
3. **Model Training** — Trains the CNN with `Adam` optimizer, `EarlyStopping`, and `ModelCheckpoint` callbacks.
4. **Model Evaluation** — Evaluates test accuracy (~85–90%), generates a Confusion Matrix heatmap, and outputs a Classification Report.
5. **TFLite Conversion** — Converts the Keras model (`.keras`) into an optimized mobile format (`model.tflite`).
6. **Android Application** — A clean, beginner-friendly Android app built with Kotlin, ViewBinding, and CameraX that allows users to take a photo or pick from the gallery to classify images instantly (`Cat` or `Dog`).

---

## ✨ Features

- 🐱 vs 🐶 Classifies images into **2 categories (`Cat` and `Dog`)**.
- 📦 Automatic zip extraction from `archive.zip` (no manual data sorting required).
- 🎨 Visualizes training progress charts, sample grids, and confusion matrices (`models/`).
- 📱 Android app with camera live capture (`CameraX`) and gallery picker support.
- 🚀 Lightweight `TensorFlow Lite` model optimized for fast mobile CPUs.
- 📝 Extensively commented, beginner-friendly code designed for clarity and interviews.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3** | Core language for the ML training pipeline |
| **TensorFlow / Keras** | Building, training, and evaluating the CNN model |
| **TensorFlow Lite** | Converting the model for fast mobile inference |
| **NumPy & SciPy** | Numerical array operations and image transformations |
| **Matplotlib & Seaborn** | Plotting training curves and confusion matrix heatmaps |
| **scikit-learn** | Computing precision, recall, and F1-score classification reports |
| **Pillow & OpenCV** | Image loading and pre-processing utilities |
| **Kotlin** | Android app development |
| **CameraX** | Modern Android camera API for capturing photos |
| **Android Studio** | IDE for building and running the mobile application |

---

## 📁 Folder Structure

```
image_classification/
│
├── archive.zip                 # Downloaded dataset zip (contains 25,000 cat & dog images)
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation file
│
├── dataset/                    # Automatically populated when step 1 runs
│   ├── README.md               # Dataset details and structure explanation
│   ├── train/                  # Training images (~2,000 cat + 2,000 dog)
│   ├── validation/             # Validation images (~500 cat + 500 dog)
│   └── test/                   # Unseen test images (~500 cat + 500 dog)
│
├── models/                     # Generated models and charts (created during training)
│   ├── best_model.keras        # Best model weights saved by ModelCheckpoint
│   ├── final_model.keras       # Final model after training finishes
│   ├── model.tflite            # Optimized TensorFlow Lite model for Android
│   ├── labels.txt              # Class names ('cat' and 'dog')
│   ├── sample_images.png       # Sample dataset grid
│   ├── augmented_samples.png   # Data augmentation demo grid
│   ├── training_history.png    # Accuracy & loss progress chart
│   └── confusion_matrix.png   # Confusion matrix heatmap
│
├── src/                        # Python ML Pipeline Scripts (run in order 01 -> 05)
│   ├── 01_data_preparation.py  # Step 1: Extract zip, organize folders, set up augmentation
│   ├── 02_build_model.py       # Step 2: Define and inspect CNN architecture
│   ├── 03_train_model.py       # Step 3: Train model with callbacks & save weights
│   ├── 04_evaluate_model.py    # Step 4: Evaluate test set & plot confusion matrix
│   └── 05_convert_tflite.py    # Step 5: Convert model to TFLite format
│
└── android-app/                # Android Studio Kotlin Project
    ├── build.gradle.kts        # Root Gradle build config
    ├── settings.gradle.kts     # Project settings & repositories
    └── app/
        ├── build.gradle.kts    # App dependencies (CameraX, TFLite, Material 3)
        └── src/main/
            ├── AndroidManifest.xml # Camera permission declaration
            ├── assets/
            │   ├── labels.txt      # Contains 'cat' and 'dog'
            │   ├── model.tflite    # <-- Copy generated TFLite model here!
            │   └── README.md
            ├── java/com/example/imageclassifier/
            │   ├── MainActivity.kt # Camera capture, gallery picker, & UI handling
            │   └── ImageClassifier.kt # TFLite model loader & 128x128 inference wrapper
            └── res/                # XML Layouts, themes, colors, and strings
```

---

## 🚀 Step-by-Step Guide: Training the Model (Python Pipeline)

Make sure you open your terminal/command prompt and navigate to the project directory:
```cmd
cd C:\Users\HP\OneDrive\Desktop\image_classification
```

### 1. Install Dependencies
Install all required Python libraries:
```cmd
pip install -r requirements.txt
```

### 2. Step 1: Prepare & Augment Data
Run the preparation script from inside the `src/` directory:
```cmd
cd src
python 01_data_preparation.py
```
- **What it does:** Locates `archive.zip`, extracts and organizes images into `dataset/train/`, `dataset/validation/`, and `dataset/test/` folders, sets up data augmentation, and saves sample visualization grids in `models/`.

### 3. Step 2: Inspect Model Architecture
```cmd
python 02_build_model.py
```
- **What it does:** Builds the 128x128 CNN model and prints the complete layer-by-layer parameter breakdown.

### 4. Step 3: Train the CNN
```cmd
python 03_train_model.py
```
- **What it does:** Trains the neural network using real-time data augmentation. Automatically stops early if validation accuracy plateaus (`EarlyStopping`) and saves the best weights to `models/best_model.keras`.
- **Time Required:** ~10–15 minutes on standard hardware.

### 5. Step 4: Evaluate Performance
```cmd
python 04_evaluate_model.py
```
- **What it does:** Evaluates the model on unseen test images, outputs exact accuracy figures, generates a Confusion Matrix heatmap (`models/confusion_matrix.png`), and prints precision/recall statistics.

### 6. Step 5: Convert to TensorFlow Lite
```cmd
python 05_convert_tflite.py
```
- **What it does:** Converts the Keras model into an optimized `model.tflite` mobile binary (~3x smaller file size), generates `models/labels.txt`, and runs a verification check.

---

## 📱 Step-by-Step Guide: Running the Android App in Android Studio

Once your `model.tflite` file is created, follow these steps to run the application on your Android phone or emulator:

### Step 1: Copy the Trained TFLite Model
Copy the generated `model.tflite` file into the Android project's `assets/` folder:
```cmd
cd C:\Users\HP\OneDrive\Desktop\image_classification
copy models\model.tflite android-app\app\src\main\assets\model.tflite
```

> **Note:** `labels.txt` is already pre-configured inside `assets/` with `cat` and `dog`.

---

### Step 2: Open the Project in Android Studio
1. Launch **Android Studio**.
2. Click **Open** (or **File → Open** if Android Studio is already running).
3. Navigate to:
   `C:\Users\HP\OneDrive\Desktop\image_classification\android-app`
4. Select the **`android-app`** folder and click **OK**.

---

### Step 3: Wait for Gradle Sync to Complete
- Android Studio will begin **Gradle Sync** to download dependencies (`CameraX`, `TensorFlow Lite Support`, `Material Design 3`, etc.).
- Look at the bottom bar. When you see ✅ **"BUILD SUCCESSFUL"** or **"Gradle sync finished"**, you are ready.

---

### Step 4: Connect Your Android Device (Recommended)
Because this app uses the camera, running on a physical Android device is ideal:
1. On your Android phone, enable **Developer Options** (**Settings → About Phone → Tap "Build Number" 7 times**).
2. Go to **Settings → Developer Options** and turn **USB Debugging ON**.
3. Connect your phone via USB cable and tap **Allow USB Debugging** on the screen prompt.
4. Your phone name will appear in the top device selection dropdown in Android Studio.

> **Using an Emulator Instead?**
> If using an Android Emulator, make sure it is targeting **API 24 or higher** (Android 7.0+). You can use the **"🖼️ Pick from Gallery"** option if emulator camera support is limited.

---

### Step 5: Build and Run the App
1. Click the green **▶️ Run** button in the top toolbar (or press `Shift + F10`).
2. Android Studio will build the APK and install it onto your phone.
3. Once opened, grant **Camera Permission** when prompted.
4. Tap **"📷 Take Photo"** to capture a cat/dog or **"🖼️ Pick from Gallery"** to select an existing photo.
5. The result display at the bottom will instantly show the prediction: **`Cat`** or **`Dog`**.

---

## 🧠 Key Concepts Explained

### Convolutional Neural Network (CNN)
A specialized neural network for vision tasks. `Conv2D` layers act like magnifying glasses scanning across the image to find edges, textures, and shapes (`cat ears`, `dog snouts`), while `MaxPooling2D` layers compress spatial dimensions to focus on the most important features.

### Data Augmentation
To prevent the network from memorizing exact training photos (`overfitting`), data augmentation randomly rotates, zooms, shifts, and flips images during training. This trains the model to recognize cats and dogs regardless of angle or lighting.

### TensorFlow Lite (.tflite)
Mobile devices have limited battery, memory, and processing power compared to servers. `TensorFlow Lite` shrinks the model weights using post-training quantization and removes heavy training code, allowing fast, offline inference directly on smartphones.
