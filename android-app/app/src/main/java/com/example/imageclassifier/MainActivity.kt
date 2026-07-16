package com.example.imageclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.imageclassifier.databinding.ActivityMainBinding

/**
 * MainActivity — The main (and only) screen of the Image Classifier app.
 *
 * This activity allows the user to:
 *   1. Take a photo using the device camera
 *   2. Pick an image from the gallery
 *   3. See the predicted CIFAR-10 class name for the selected image
 *
 * It uses:
 *   - ViewBinding for type-safe view access (no findViewById needed)
 *   - ActivityResultContracts for modern camera/gallery interaction
 *   - ImageClassifier helper class for TFLite inference
 *
 * The classification result shows ONLY the class name (e.g., "Cat"),
 * without any confidence score.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    // ======================================================================
    // Properties
    // ======================================================================

    // ViewBinding instance — gives us direct access to all views in the layout
    // without calling findViewById(). Generated from activity_main.xml.
    private lateinit var binding: ActivityMainBinding

    // Our TFLite image classifier helper
    // Initialized lazily (only when first needed) to avoid slowing down app startup
    private var classifier: ImageClassifier? = null

    // ======================================================================
    // Activity Result Launchers
    // ======================================================================
    // These are the modern replacement for startActivityForResult().
    // They are registered here (as properties) so they are ready before onCreate.

    /**
     * Camera permission request launcher.
     * When the user responds to the permission dialog, this callback fires.
     * If granted, we launch the camera. If denied, we show a toast message.
     */
    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            // Permission was granted — go ahead and launch the camera
            Log.d(TAG, "Camera permission granted")
            takePictureLauncher.launch(null)
        } else {
            // Permission was denied — inform the user
            Log.d(TAG, "Camera permission denied")
            Toast.makeText(
                this,
                getString(R.string.camera_permission_denied),
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    /**
     * Camera capture launcher.
     * Uses TakePicturePreview() which returns a Bitmap thumbnail directly.
     * This is simpler than TakePicture() which requires a Uri and file provider.
     * The returned bitmap is suitable for CIFAR-10 (which only needs 32x32 anyway).
     */
    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap: Bitmap? ->
        // The camera may return null if the user cancelled
        if (bitmap != null) {
            Log.d(TAG, "Photo captured: ${bitmap.width}x${bitmap.height}")
            processImage(bitmap)
        } else {
            Log.d(TAG, "Camera capture cancelled")
        }
    }

    /**
     * Gallery image picker launcher.
     * Uses GetContent() with "image/any" MIME type to let the user pick any image.
     * Returns a Uri pointing to the selected image.
     */
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        // The picker may return null if the user cancelled
        if (uri != null) {
            Log.d(TAG, "Image selected from gallery: $uri")
            // Convert the Uri to a Bitmap
            val bitmap = uriToBitmap(uri)
            if (bitmap != null) {
                processImage(bitmap)
            } else {
                Log.e(TAG, "Failed to decode image from URI")
                binding.resultText.text = getString(R.string.error_classification)
            }
        } else {
            Log.d(TAG, "Gallery picker cancelled")
        }
    }

    // ======================================================================
    // Lifecycle Methods
    // ======================================================================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // ------------------------------------------------------------------
        // Step 1: Set up ViewBinding
        // ------------------------------------------------------------------
        // ViewBinding generates a binding class from the XML layout.
        // It provides direct references to all views with IDs.
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // ------------------------------------------------------------------
        // Step 2: Set up button click listeners
        // ------------------------------------------------------------------

        // "Take Photo" button — launches the camera
        binding.btnTakePhoto.setOnClickListener {
            handleTakePhoto()
        }

        // "Pick from Gallery" button — opens the gallery picker
        binding.btnPickGallery.setOnClickListener {
            handlePickFromGallery()
        }

        Log.d(TAG, "MainActivity created successfully")
    }

    override fun onDestroy() {
        super.onDestroy()

        // ------------------------------------------------------------------
        // Clean up: Release the TFLite interpreter
        // ------------------------------------------------------------------
        // This frees the native memory used by the TFLite model.
        // Always close the classifier when the activity is destroyed.
        classifier?.close()
        classifier = null
        Log.d(TAG, "Classifier released")
    }

    // ======================================================================
    // Button Handlers
    // ======================================================================

    /**
     * Handle the "Take Photo" button click.
     *
     * Before launching the camera, we need to check if the CAMERA permission
     * has been granted. Android requires runtime permissions for the camera.
     *
     * Flow:
     *   1. Check if permission is already granted
     *   2. If yes → launch camera directly
     *   3. If no → request permission (the result callback will launch camera if granted)
     */
    private fun handleTakePhoto() {
        when {
            // Check if we already have camera permission
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                // Permission already granted — launch camera
                Log.d(TAG, "Camera permission already granted, launching camera")
                takePictureLauncher.launch(null)
            }

            else -> {
                // Need to request permission
                Log.d(TAG, "Requesting camera permission")
                requestCameraPermission.launch(Manifest.permission.CAMERA)
            }
        }
    }

    /**
     * Handle the "Pick from Gallery" button click.
     *
     * No special permissions needed for the gallery picker on modern Android.
     * The GetContent() contract handles everything for us.
     */
    private fun handlePickFromGallery() {
        Log.d(TAG, "Opening gallery picker")
        pickImageLauncher.launch("image/*")
    }

    // ======================================================================
    // Image Processing
    // ======================================================================

    /**
     * Process a captured/selected image:
     *   1. Display it in the ImageView
     *   2. Run classification
     *   3. Display the result
     *
     * @param bitmap  The image to classify
     */
    private fun processImage(bitmap: Bitmap) {
        // ------------------------------------------------------------------
        // Step 1: Display the image in the ImageView
        // ------------------------------------------------------------------
        binding.imageView.setImageBitmap(bitmap)

        // Show a temporary "Classifying..." message while processing
        binding.resultText.text = getString(R.string.classifying)

        // ------------------------------------------------------------------
        // Step 2: Run classification
        // ------------------------------------------------------------------
        try {
            // Initialize the classifier if it hasn't been created yet
            // We do this lazily to avoid slowing down app startup
            if (classifier == null) {
                classifier = ImageClassifier(this)
            }

            // Run inference — this returns Pair(className, confidence) e.g., Pair("Cat", 0.99f)
            val (result, confidence) = classifier!!.classify(bitmap)

            // Option B: Check if model predicted the 3rd class ('Other')
            val displayText = if (result.equals("Other", ignoreCase = true)) {
                "Other (Not Cat/Dog)"
            } else {
                result
            }
            binding.resultText.text = displayText
            Log.d(TAG, "Classification complete: $displayText (confidence = ${confidence * 100}%)")

        } catch (e: Exception) {
            // ------------------------------------------------------------------
            // Error handling
            // ------------------------------------------------------------------
            // If anything goes wrong (model not found, inference error, etc.),
            // show an error message and log the full exception
            Log.e(TAG, "Classification failed", e)
            binding.resultText.text = getString(R.string.error_classification)
            Toast.makeText(
                this,
                "Classification error: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    /**
     * Convert a content Uri to a Bitmap.
     *
     * Uses different methods depending on the Android version:
     *   - API 28+ (Pie): Uses ImageDecoder (newer, better)
     *   - API < 28: Uses MediaStore.Images (older, deprecated but functional)
     *
     * @param uri  The content URI of the image
     * @return     A Bitmap, or null if decoding failed
     */
    private fun uriToBitmap(uri: Uri): Bitmap? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                // Modern approach (Android 9+)
                // ImageDecoder is more efficient and supports more formats
                val source = ImageDecoder.createSource(contentResolver, uri)
                // We request a mutable, software-rendered bitmap for TFLite compatibility
                ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    decoder.isMutableRequired = true
                }
            } else {
                // Legacy approach (Android 7-8)
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to decode bitmap from URI: $uri", e)
            null
        }
    }
}
