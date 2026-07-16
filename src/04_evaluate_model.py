"""
=============================================================================
 Step 4: Evaluate the Trained CNN Model (Cats vs Dogs)
=============================================================================

 PURPOSE:
 This script evaluates our trained CNN model on unseen test images from
 `archive.zip` (`dataset/test/`) and generates detailed performance metrics.

 WHAT THIS SCRIPT DOES:
 1. Loads the best saved model (`models/best_model.keras`).
 2. Evaluates overall accuracy on the test dataset using `.evaluate()`.
 3. Generates predictions across all test images using `.predict()`.
 4. Computes and visualizes a Confusion Matrix (`models/confusion_matrix.png`).
 5. Prints a detailed Classification Report (Precision, Recall, F1-Score).
 6. Explains what every metric means in plain, beginner-friendly language.

 PREREQUISITES:
 - Must have trained the model using `03_train_model.py`.

 OUTPUTS:
 - Printed accuracy and classification metrics.
 - ../models/confusion_matrix.png (Heatmap of true vs predicted classes)

=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

import importlib.util
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

data_prep = importlib.import_module("01_data_preparation")
MODELS_DIR = data_prep.MODELS_DIR


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Creates a Seaborn heatmap showing the Confusion Matrix and saves
    it to `models/confusion_matrix.png`.
    """
    print("\n[INFO] Generating Confusion Matrix heatmap...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[c.capitalize() for c in classes],
        yticklabels=[c.capitalize() for c in classes],
        cbar=False,
        annot_kws={"size": 16, "weight": "bold"}
    )
    plt.title('Confusion Matrix (Cats vs Dogs)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Predicted Class', fontsize=13, fontweight='bold')
    plt.ylabel('True Class', fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Saved Confusion Matrix chart to -> {save_path}")


def main():
    print("=====================================================================")
    print(" [Cats vs Dogs] Step 4: Evaluate Model on Unseen Test Dataset")
    print("=====================================================================\n")

    # Step 1: Locate and load the best saved model
    best_model_path = os.path.join(MODELS_DIR, "best_model.keras")
    if not os.path.exists(best_model_path):
        # Fallback to final_model.keras
        fallback_path = os.path.join(MODELS_DIR, "final_model.keras")
        if os.path.exists(fallback_path):
            best_model_path = fallback_path
        else:
            raise FileNotFoundError(
                f"\n[ERROR] Could not find saved model at:\n  {best_model_path}\n"
                "Please run `python 03_train_model.py` first to train the model."
            )

    print(f"[INFO] Loading trained model from: {best_model_path}")
    model = tf.keras.models.load_model(best_model_path)

    # Step 2: Load test data generator (shuffle=False is required to match predictions with filenames/labels!)
    _, _, test_generator = data_prep.get_data_generators()

    # Step 3: Evaluate overall accuracy
    print("\n[INFO] Running evaluation on unseen test images...")
    loss, accuracy = model.evaluate(test_generator, verbose=1)
    print(f"\n=====================================================================")
    print(f" OVERALL TEST ACCURACY: {accuracy * 100:.2f}%")
    print(f" OVERALL TEST LOSS:     {loss:.4f}")
    print(f"=====================================================================")

    # Step 4: Generate predictions for all test samples
    print("\n[INFO] Generating predictions for confusion matrix & report...")
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Step 5: Plot Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, data_prep.CLASSES)

    # Step 6: Print Classification Report
    print("\n--- Detailed Classification Report ---")
    report = classification_report(
        y_true,
        y_pred,
        target_names=[c.capitalize() for c in data_prep.CLASSES]
    )
    print(report)

    # Step 7: Print Beginner-Friendly Metric Explanations
    print("\n--- What Do These Metrics Mean? ---")
    print(" * Accuracy:        The percentage of total test images correctly classified.")
    print(" * Confusion Matrix: Shows exactly where the model got confused.")
    print("                     - Top-Left: True Cats correctly predicted as Cat.")
    print("                     - Bottom-Right: True Dogs correctly predicted as Dog.")
    print("                     - Off-diagonals are mistakes (e.g., Cat misclassified as Dog).")
    print(" * Precision:       When the model predicts 'Cat', how often is it actually correct?")
    print(" * Recall:          Out of all real Cats in the test set, what percentage did the model find?")
    print(" * F1-Score:        The harmonic balance between Precision and Recall.")

    print("\n=====================================================================")
    print(" [OK] Step 4 Complete! You are now ready to run `05_convert_tflite.py`.")
    print("=====================================================================")


if __name__ == "__main__":
    main()
