package com.example.imageclassifier

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * ImageClassifier — A helper class that wraps a TensorFlow Lite model
 * for CIFAR-10 image classification.
 *
 * This class handles:
 *   1. Loading the TFLite model from the app's assets folder
 *   2. Loading the class labels from a text file
 *   3. Pre-processing input images (resize + normalize)
 *   4. Running inference and returning the predicted class name
 *
 * Usage:
 *   val classifier = ImageClassifier(context)
 *   val result = classifier.classify(bitmap)   // Returns e.g. "Cat"
 *   classifier.close()                          // Release resources when done
 *
 * @param context  Android Context used to access the assets folder
 */
class ImageClassifier(context: Context) {

    companion object {
        private const val TAG = "ImageClassifier"

        // Cats vs Dogs images are 128x128 pixels with 3 color channels (RGB)
        private const val IMAGE_SIZE = 128
        private const val NUM_CHANNELS = 3

        // Each pixel channel is a 32-bit float (4 bytes)
        private const val BYTES_PER_CHANNEL = 4

        // The model file and label file names in the assets folder
        private const val MODEL_FILE = "model.tflite"
        private const val LABELS_FILE = "labels.txt"
    }

    // The TensorFlow Lite interpreter that runs the model
    private var interpreter: Interpreter

    // List of class labels loaded from labels.txt
    // e.g., ["airplane", "automobile", "bird", "cat", ...]
    private var labels: List<String>

    init {
        // ------------------------------------------------------------------
        // Step 1: Load the TFLite model from assets
        // ------------------------------------------------------------------
        // We memory-map the model file for efficient loading.
        // This avoids copying the entire model into RAM.
        val model = loadModelFile(context)
        interpreter = Interpreter(model)
        Log.d(TAG, "TFLite model loaded successfully")

        // ------------------------------------------------------------------
        // Step 2: Load the class labels from assets
        // ------------------------------------------------------------------
        labels = loadLabels(context)
        Log.d(TAG, "Loaded ${labels.size} labels: $labels")
    }

    /**
     * Classify a bitmap image and return the predicted class name.
     *
     * The method performs these steps:
     *   1. Resize the bitmap to 128x128 (model input size)
     *   2. Convert pixel values to a normalized float ByteBuffer [0, 1]
     *   3. Run the TFLite model inference
     *   4. Find the class with the highest probability
     *   5. Return the class name (e.g., "Cat")
     *
     * @param bitmap  The input image to classify (any size)
     * @return        The predicted class name with the first letter capitalized
     */
    fun classify(bitmap: Bitmap): Pair<String, Float> {
        // ------------------------------------------------------------------
        // Step 1: Resize the image to 128x128 pixels
        // ------------------------------------------------------------------
        // We use bilinear filtering (enabled by the `true` parameter)
        // for smoother downscaling.
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)

        // ------------------------------------------------------------------
        // Step 2: Convert the bitmap to a normalized float ByteBuffer
        // ------------------------------------------------------------------
        // The model expects input as a flat array of floats:
        //   Shape: [1, 128, 128, 3]  (batch, height, width, channels)
        //   Values: normalized to [0.0, 1.0] range
        //
        // Total bytes = 1 × 128 × 128 × 3 × 4 = 196,608 bytes
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

        // ------------------------------------------------------------------
        // Step 3: Prepare the output array
        // ------------------------------------------------------------------
        // The model outputs a probability distribution over 10 classes.
        // Shape: [1, 10] — one probability for each CIFAR-10 class
        val output = Array(1) { FloatArray(labels.size) }

        // ------------------------------------------------------------------
        // Step 4: Run inference
        // ------------------------------------------------------------------
        // The interpreter takes the input buffer and fills the output array
        // with probability scores for each class.
        interpreter.run(inputBuffer, output)

        // ------------------------------------------------------------------
        // Step 5: Find the class with the highest probability
        // ------------------------------------------------------------------
        val probabilities = output[0]
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val maxConfidence = probabilities[maxIndex]

        // Return the class name with the first letter capitalized
        // e.g., "cat" → "Cat", "airplane" → "Airplane"
        val className = labels[maxIndex]
        val formattedName = className.replaceFirstChar { it.uppercase() }

        Log.d(TAG, "Classification result: $formattedName (confidence=${maxConfidence * 100}%, index=$maxIndex)")
        return Pair(formattedName, maxConfidence)
    }

    /**
     * Release the TFLite interpreter resources.
     * Call this when the classifier is no longer needed (e.g., in onDestroy).
     */
    fun close() {
        interpreter.close()
        Log.d(TAG, "Interpreter closed")
    }

    // ======================================================================
    // Private helper methods
    // ======================================================================

    /**
     * Load the TFLite model file from the assets folder using memory mapping.
     *
     * Memory mapping is preferred because:
     *   - It doesn't copy the entire model into RAM
     *   - The OS can page parts of the model in/out as needed
     *   - It's faster for large model files
     *
     * @param context  Android Context for accessing assets
     * @return         A MappedByteBuffer containing the model data
     */
    private fun loadModelFile(context: Context): MappedByteBuffer {
        // Open the model file from the assets folder
        val fileDescriptor = context.assets.openFd(MODEL_FILE)

        // Create an input stream and get the file channel
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel

        // Map the model file into memory
        // startOffset and length ensure we read only the model file
        // (assets can be packed together in the APK)
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength

        val mappedBuffer = fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        )

        // Clean up streams (the mapped buffer remains valid)
        fileChannel.close()
        inputStream.close()
        fileDescriptor.close()

        return mappedBuffer
    }

    /**
     * Load class labels from a text file in the assets folder.
     *
     * Each line in the file should contain one label.
     * Example content of labels.txt:
     *   airplane
     *   automobile
     *   bird
     *   ...
     *
     * @param context  Android Context for accessing assets
     * @return         A list of label strings
     */
    private fun loadLabels(context: Context): List<String> {
        val labels = mutableListOf<String>()

        // Open the labels file and read line by line
        context.assets.open(LABELS_FILE).use { inputStream ->
            BufferedReader(InputStreamReader(inputStream)).use { reader ->
                var line = reader.readLine()
                while (line != null) {
                    // Only add non-blank lines
                    if (line.isNotBlank()) {
                        labels.add(line.trim())
                    }
                    line = reader.readLine()
                }
            }
        }

        return labels
    }

    /**
     * Convert a Bitmap to a ByteBuffer suitable for TFLite input.
     *
     * This method:
     *   1. Extracts pixel values from the bitmap
     *   2. Separates R, G, B channels
     *   3. Normalizes each channel value from [0, 255] to [0.0, 1.0]
     *   4. Packs them into a ByteBuffer in the order expected by the model
     *
     * The resulting ByteBuffer has shape [1, 32, 32, 3] when interpreted
     * as a tensor of floats.
     *
     * @param bitmap  A 32x32 bitmap (should already be resized)
     * @return        A ByteBuffer ready for TFLite inference
     */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        // Calculate total buffer size:
        // 1 batch × 32 height × 32 width × 3 channels × 4 bytes per float
        val bufferSize = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS * BYTES_PER_CHANNEL

        // Allocate a direct ByteBuffer (required by TFLite)
        // Direct buffers live outside the JVM heap for better performance
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        // Extract all pixel values from the bitmap into an IntArray
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        bitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        // Iterate through each pixel and extract RGB channels
        for (pixel in pixels) {
            // Extract Red channel (bits 16-23), normalize to [0, 1]
            val r = ((pixel shr 16) and 0xFF) / 255.0f

            // Extract Green channel (bits 8-15), normalize to [0, 1]
            val g = ((pixel shr 8) and 0xFF) / 255.0f

            // Extract Blue channel (bits 0-7), normalize to [0, 1]
            val b = (pixel and 0xFF) / 255.0f

            // Write the normalized values to the buffer
            // Order: R, G, B for each pixel (channel-last format)
            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }

        return byteBuffer
    }
}
