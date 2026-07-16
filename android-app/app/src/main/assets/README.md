# Android App Assets Folder

After converting your model using `python 05_convert_tflite.py`, copy the `model.tflite` file into this folder before running the app in Android Studio:

```cmd
copy ..\..\..\..\..\models\model.tflite .
```

Or from the root project directory (`C:\Users\HP\OneDrive\Desktop\image_classification`):
```cmd
copy models\model.tflite android-app\app\src\main\assets\model.tflite
```

This folder should contain:
- `model.tflite` (Your trained Cats vs Dogs TFLite model)
- `labels.txt` (Contains `cat` and `dog`)
