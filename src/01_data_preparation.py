"""
=============================================================================
 Step 1: Data Preparation for CNN Image Classification (Cats vs Dogs)
=============================================================================

 PURPOSE:
 This script extracts images from the local `archive.zip` dataset, organizes
 them into train/validation/test folders, applies normalization & augmentation,
 and visualizes sample images.

 WHAT THIS SCRIPT DOES:
 1. Locates `archive.zip` in the project root directory.
 2. Extracts and organizes images into:
    - ../dataset/train/cat/       and  ../dataset/train/dog/
    - ../dataset/validation/cat/  and  ../dataset/validation/dog/
    - ../dataset/test/cat/        and  ../dataset/test/dog/
 3. Normalizes pixel values from [0-255] to [0-1] range using ImageDataGenerator.
 4. Sets up Data Augmentation (creates variations of training images).
 5. Visualizes sample training images and augmented images.

 PREREQUISITES:
 - pip install -r ../requirements.txt
 - Place `archive.zip` in the project root directory: C:\\Users\\HP\\OneDrive\\Desktop\\image_classification\\archive.zip

 OUTPUTS:
 - Extracted folders under `../dataset/`
 - ../models/sample_images.png     (grid of 9 sample images)
 - ../models/augmented_samples.png (grid of 9 augmented variations of one image)

=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import zipfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ── Configuration Constants ──────────────────────────────────────────────────
# We use __file__ to locate paths relative to this script's directory (src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ZIP_PATH = os.path.join(PROJECT_ROOT, "archive.zip")

# Ensure the models directory exists for saving plots
os.makedirs(MODELS_DIR, exist_ok=True)

# Image dimensions required for our CNN model (Cats vs Dogs images are larger than CIFAR-10)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
CLASSES = ["cat", "dog", "other"]

# Number of images to extract per class for each split.
# Using 3000 train + 500 val + 500 test per class (9,000 images total) allows robust training while remaining fast.
# Set to None if you want to use all 12,500 images per class from the zip file.
NUM_TRAIN_PER_CLASS = 3000
NUM_VAL_PER_CLASS = 500
NUM_TEST_PER_CLASS = 500


def extract_and_organize_dataset(force_reextract=False):
    """
    Extracts images from `archive.zip` and organizes them into train/validation/test
    subdirectories for seamless use with Keras flow_from_directory().
    """
    if force_reextract and os.path.exists(DATASET_DIR):
        print(f"[INFO] force_reextract=True -> Cleaning existing dataset directory: {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR, ignore_errors=True)

    train_cat_dir = os.path.join(DATASET_DIR, "train", "cat")
    train_dog_dir = os.path.join(DATASET_DIR, "train", "dog")

    # Check if images are already extracted and organized
    if not force_reextract and os.path.exists(train_cat_dir) and len(os.listdir(train_cat_dir)) > 0:
        print("[OK] Dataset already extracted and organized under:")
        print(f"   {DATASET_DIR}")
        return

    print("[INFO] Locating archive.zip dataset...")
    if not os.path.exists(ZIP_PATH):
        # Check fallback path just in case
        fallback_zip = os.path.join(DATASET_DIR, "archive.zip")
        if os.path.exists(fallback_zip):
            zip_to_open = fallback_zip
        else:
            raise FileNotFoundError(
                f"\n[ERROR] Could not find 'archive.zip' at:\n  {ZIP_PATH}\n"
                "Please ensure archive.zip is placed directly inside the project folder."
            )
    else:
        zip_to_open = ZIP_PATH

    print(f"[INFO] Opening zip archive: {zip_to_open}")
    with zipfile.ZipFile(zip_to_open, "r") as z:
        all_files = z.namelist()
        
        # Filter all cat and dog jpg images from the zip entries
        cat_files = sorted([f for f in all_files if os.path.basename(f).startswith("cat.") and f.endswith(".jpg")])
        dog_files = sorted([f for f in all_files if os.path.basename(f).startswith("dog.") and f.endswith(".jpg")])
        
        print(f"   Found {len(cat_files)} cat images and {len(dog_files)} dog images inside zip.")

        # Collect background/other images from caltech-101 if available
        caltech_dir = os.path.join(PROJECT_ROOT, "caltech-101", "101_ObjectCategories")
        other_files = []
        if os.path.exists(caltech_dir):
            import glob
            exclude = ['cat', 'dog', 'dalmatian', 'cougar_body', 'cougar_face', 'leopard', 'BACKGROUND_Google']
            folders = [d for d in os.listdir(caltech_dir) if os.path.isdir(os.path.join(caltech_dir, d)) and d not in exclude]
            for f in sorted(folders):
                other_files.extend(sorted(glob.glob(os.path.join(caltech_dir, f, '*.jpg'))))
        print(f"   Found {len(other_files)} background/other images in caltech-101.")

        # Define splits for all classes
        splits = {
            "cat": cat_files,
            "dog": dog_files,
            "other": other_files
        }

        for class_name, file_list in splits.items():
            # Determine split boundaries
            n_train = NUM_TRAIN_PER_CLASS if NUM_TRAIN_PER_CLASS else int(len(file_list) * 0.8)
            n_val = NUM_VAL_PER_CLASS if NUM_VAL_PER_CLASS else int(len(file_list) * 0.1)
            n_test = NUM_TEST_PER_CLASS if NUM_TEST_PER_CLASS else len(file_list) - n_train - n_val

            train_files = file_list[:n_train]
            val_files = file_list[n_train : n_train + n_val]
            test_files = file_list[n_train + n_val : n_train + n_val + n_test]

            split_map = {
                "train": train_files,
                "validation": val_files,
                "test": test_files
            }

            for split_name, files in split_map.items():
                target_dir = os.path.join(DATASET_DIR, split_name, class_name)
                os.makedirs(target_dir, exist_ok=True)
                print(f"   Writing {len(files)} {class_name} images to -> {split_name}/{class_name}/")
                for f_path in files:
                    if class_name == "other":
                        parent_folder = os.path.basename(os.path.dirname(f_path))
                        filename = f"{parent_folder}_{os.path.basename(f_path)}"
                        dest_file = os.path.join(target_dir, filename)
                        if not os.path.exists(dest_file):
                            shutil.copy2(f_path, dest_file)
                    else:
                        filename = os.path.basename(f_path)
                        dest_file = os.path.join(target_dir, filename)
                        if not os.path.exists(dest_file):
                            with z.open(f_path) as source_file, open(dest_file, "wb") as target_file:
                                shutil.copyfileobj(source_file, target_file)

    print("[OK] Dataset successfully extracted and organized!")


def get_data_generators():
    """
    Creates and returns Keras ImageDataGenerators for training, validation, and testing.
    
    1. Rescale: Divides all pixel values by 255.0 so they fall between 0.0 and 1.0.
    2. Augmentation (Training only): Randomly rotates, shifts, zooms, and flips images
       to help the model generalize better and avoid overfitting.
    """
    # Training generator WITH Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,          # Randomly rotate images up to 20 degrees
        width_shift_range=0.2,      # Randomly shift images horizontally up to 20%
        height_shift_range=0.2,     # Randomly shift images vertically up to 20%
        shear_range=0.2,            # Randomly apply shear transformation
        zoom_range=0.2,             # Randomly zoom inside images up to 20%
        horizontal_flip=True,       # Randomly flip images horizontally
        fill_mode="nearest"         # Fill newly created blank pixels after rotation/shifting
    )

    # Validation and Test generators ONLY rescale (no augmentation during evaluation!)
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "validation")
    test_dir = os.path.join(DATASET_DIR, "test")

    print("\n--- Loading Training Data Generator ---")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",   # Categorical for multi-output softmax (cat=0, dog=1)
        shuffle=True
    )

    print("\n--- Loading Validation Data Generator ---")
    val_generator = eval_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    print("\n--- Loading Test Data Generator ---")
    test_generator = eval_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def visualize_sample_images(train_generator):
    """
    Displays 9 sample images from the training set in a 3x3 grid
    and saves the plot to `models/sample_images.png`.
    """
    print("\n[INFO] Generating sample images grid...")
    # Grab one batch of images and one-hot labels from the generator
    images, labels = next(train_generator)
    
    # Class indices mapping: { 'cat': 0, 'dog': 1 }
    idx_to_class = {v: k.capitalize() for k, v in train_generator.class_indices.items()}

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        class_idx = np.argmax(labels[i])
        plt.title(f"Class: {idx_to_class[class_idx]}", fontsize=14, fontweight="bold")
        plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(MODELS_DIR, "sample_images.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved sample images to -> {save_path}")


def visualize_augmented_samples():
    """
    Demonstrates Data Augmentation by taking a single cat image from the dataset,
    generating 9 augmented variations, and saving to `models/augmented_samples.png`.
    """
    print("\n[INFO] Generating augmented variations demo...")
    # Find a sample cat image from the training set
    sample_dir = os.path.join(DATASET_DIR, "train", "cat")
    sample_files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(".jpg")]
    
    if not sample_files:
        print("   No sample images found to demonstrate augmentation.")
        return

    sample_img_path = sample_files[0]
    img = load_img(sample_img_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 128, 128, 3)

    # Augmentation generator specifically for demo
    demo_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    plt.figure(figsize=(10, 10))
    # Generate 9 augmented versions
    aug_iter = demo_datagen.flow(img_array, batch_size=1)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        augmented_img = next(aug_iter)[0]
        plt.imshow(augmented_img)
        plt.title(f"Variation {i + 1}", fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(MODELS_DIR, "augmented_samples.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved augmented samples to -> {save_path}")


def main():
    print("=====================================================================")
    print(" [Cats vs Dogs] Step 1: Data Preparation & Augmentation (archive.zip)")
    print("=====================================================================\n")

    # Step 1: Extract and organize images from archive.zip (force_reextract=True to update split size)
    extract_and_organize_dataset(force_reextract=True)

    # Step 2: Create data generators
    train_gen, val_gen, test_gen = get_data_generators()

    # Step 3: Print summary
    print("\n--- Dataset Summary ---")
    print(f"   Training images:   {train_gen.samples}")
    print(f"   Validation images: {val_gen.samples}")
    print(f"   Test images:       {test_gen.samples}")
    print(f"   Image shape:       {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} x 3 (RGB)")
    print(f"   Classes mapping:   {train_gen.class_indices}")

    # Step 4: Visualize sample images and augmented variations
    visualize_sample_images(train_gen)
    visualize_augmented_samples()

    print("\n=====================================================================")
    print(" [OK] Step 1 Complete! You are now ready to run `02_build_model.py`.")
    print("=====================================================================")


if __name__ == "__main__":
    main()
