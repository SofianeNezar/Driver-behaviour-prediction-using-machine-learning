// Author: Sofiane Nezar
// GitHub: https://github.com/SofianeNezar

// MainActivity.kt
package com.example.motionclassifier

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import ai.onnxruntime.*
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.FileWriter
import java.io.IOException
import java.io.InputStreamReader
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.max
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Divider
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.TextButton
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.flow.collectLatest
import com.example.motionclassifier.PredictionEntity
import com.example.motionclassifier.PredictionDao
import com.example.motionclassifier.PredictionDatabase
import kotlinx.coroutines.delay

class MainActivity : ComponentActivity(), SensorEventListener {

    companion object {
        private const val TAG = "Classificateur de conduite"
        private const val TARGET_SAMPLES = 700
        private const val SAMPLING_RATE_HZ = 100
        private const val SAMPLING_INTERVAL_MS = 1000L / SAMPLING_RATE_HZ
    }

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private val modelManager = ModelManager()

    private val sensorData = mutableListOf<List<Double>>()
    private var lastAccelData: FloatArray? = null
    private var lastGyroData: FloatArray? = null
    private var samplingTimer: Timer? = null
    private var isCollecting = false

    private var collectedSamples by mutableStateOf(0)
    private var prediction by mutableStateOf("No prediction")
    private var confidence by mutableStateOf(0.0)
    private var isModelLoading by mutableStateOf(true)
    private var modelStatus by mutableStateOf("Initializing...")
    private var lastAccelDisplay by mutableStateOf(FloatArray(3))
    private var lastGyroDisplay by mutableStateOf(FloatArray(3))
    private var isExporting by mutableStateOf(false)
    private var selectedCombinedFile by mutableStateOf<Uri?>(null)

    // CSV Upload
    private var isProcessingCSV by mutableStateOf(false)
    private var csvUploadStatus by mutableStateOf("")
    private var selectedAccelFile by mutableStateOf<Uri?>(null)
    private var selectedGyroFile by mutableStateOf<Uri?>(null)

    // Auto-prediction
    private var autoPredictTimer: Timer? = null
    private var isAutoPredicting by mutableStateOf(false)

    // Bdd
    private lateinit var database: PredictionDatabase
    private var predictionHistory by mutableStateOf(listOf<PredictionEntity>())
    private var showHistoryDialog by mutableStateOf(false)


    private var showGraphDialog by mutableStateOf(false)
    private var allCollectedData = mutableListOf<List<Double>>() // Pour stocker toutes les données
    private var isRecordingSession by mutableStateOf(false)
    private val sessionData = mutableListOf<List<Double>>()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            initializeSensors()
        } else {
            Toast.makeText(this, "Sensor permission required", Toast.LENGTH_LONG).show()
            modelStatus = "Permission denied - sensors unavailable"
            isModelLoading = false
        }
    }


    private val pickAccelCsvLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            selectedAccelFile = it
            csvUploadStatus = "Accelerometer file selected: ${getFileName(it)}"
            Log.d(TAG, "Accelerometer CSV selected: $it")
        }
    }


    private val pickGyroCsvLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            selectedGyroFile = it
            csvUploadStatus = "Gyroscope file selected: ${getFileName(it)}"
            Log.d(TAG, "Gyroscope CSV selected: $it")
        }
    }
    private val pickCombinedCsvLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            selectedCombinedFile = it
            csvUploadStatus = "Combined file selected: ${getFileName(it)}"
            Log.d(TAG, "Combined CSV selected: $it")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "onCreate() starting")

        try {
            sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

            setContent {
                MaterialTheme {
                    MotionClassifierScreen()
                }
            }

            checkPermissionAndInitialize()

            // Initialiser base de données
            database = PredictionDatabase.getDatabase(this)

            // Charger historique
            loadPredictionHistory()
        } catch (e: Exception) {
            Log.e(TAG, "Error in onCreate", e)
            Toast.makeText(this, "App initialization failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun getFileName(uri: Uri): String {
        return uri.lastPathSegment ?: "unknown_file.csv"
    }

    private fun checkPermissionAndInitialize() {
        try {
            when {
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.BODY_SENSORS
                ) == PackageManager.PERMISSION_GRANTED -> {
                    Log.d(TAG, "BODY_SENSORS permission granted")
                    initializeSensors()
                }
                else -> {
                    Log.d(TAG, "Requesting BODY_SENSORS permission")
                    initializeSensors()
                    requestPermissionLauncher.launch(Manifest.permission.BODY_SENSORS)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error checking permissions", e)
            modelStatus = "Permission check failed: ${e.message}"
            isModelLoading = false
        }
    }

    private fun initializeSensors() {
        try {
            Log.d(TAG, "Initializing sensors...")

            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
            gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

            Log.d(TAG, "Accelerometer available: ${accelerometer != null}")
            Log.d(TAG, "Gyroscope available: ${gyroscope != null}")

            if (accelerometer == null && gyroscope == null) {
                Toast.makeText(this, "No motion sensors available on this device", Toast.LENGTH_LONG).show()
                modelStatus = "No sensors available"
                isModelLoading = false
                return
            }

            if (accelerometer == null) {
                Toast.makeText(this, "Accelerometer not available", Toast.LENGTH_LONG).show()
                modelStatus = "Accelerometer missing"
                isModelLoading = false
                return
            }

            if (gyroscope == null) {
                Toast.makeText(this, "Gyroscope not available", Toast.LENGTH_LONG).show()
                modelStatus = "Gyroscope missing"
                isModelLoading = false
                return
            }

            // Configuration à 100Hz
            val accelRegistered = sensorManager.registerListener(
                this,
                accelerometer,
                SensorManager.SENSOR_DELAY_FASTEST
            )

            val gyroRegistered = sensorManager.registerListener(
                this,
                gyroscope,
                SensorManager.SENSOR_DELAY_FASTEST
            )

            Log.d(TAG, "Accelerometer registered: $accelRegistered")
            Log.d(TAG, "Gyroscope registered: $gyroRegistered")

            if (!accelRegistered || !gyroRegistered) {
                Toast.makeText(this, "Failed to register sensor listeners", Toast.LENGTH_LONG).show()
                modelStatus = "Sensor registration failed"
                isModelLoading = false
                return
            }

            initializeModel()

        } catch (e: Exception) {
            Log.e(TAG, "Error initializing sensors", e)
            Toast.makeText(this, "Sensor initialization failed: ${e.message}", Toast.LENGTH_LONG).show()
            modelStatus = "Sensor error: ${e.message}"
            isModelLoading = false
        }
    }

    private fun initializeModel() {
        lifecycleScope.launch {
            try {
                Log.d(TAG, "Starting model initialization...")
                isModelLoading = true
                modelStatus = "Loading model resources..."

                val initialized = modelManager.initialize(this@MainActivity)

                isModelLoading = false
                if (initialized) {
                    modelStatus = "Model ready"
                    Log.d(TAG, "Model initialized successfully")
                    startSampling()
                } else {
                    modelStatus = "Model initialization failed - check if model files exist in assets/"
                    Log.e(TAG, "Model initialization failed")
                }
            } catch (e: Exception) {
                isModelLoading = false
                modelStatus = "Error: ${e.message}"
                Log.e(TAG, "Model initialization error", e)
                Toast.makeText(this@MainActivity, "Model error: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun startSampling() {
        try {
            Log.d(TAG, "Starting data sampling...")
            sensorData.clear()
            collectedSamples = 0
            isCollecting = true

            samplingTimer?.cancel()
            samplingTimer = Timer().apply {
                scheduleAtFixedRate(object : TimerTask() {
                    override fun run() {
                        try {
                            collectSample()
                        } catch (e: Exception) {
                            Log.e(TAG, "Error collecting sample", e)
                        }
                    }
                }, 0, SAMPLING_INTERVAL_MS)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error starting sampling", e)
            isCollecting = false
        }
    }

    private fun collectSample() {
        val accel = lastAccelData
        val gyro = lastGyroData

        if (accel == null || gyro == null) {
            if (collectedSamples % 100 == 0) {
                Log.d(TAG, "Waiting for sensor data... accel=${accel != null}, gyro=${gyro != null}")
            }
            return
        }

        try {
            synchronized(sensorData) {
                if (accel.any { it.isNaN() || it.isInfinite() } ||
                    gyro.any { it.isNaN() || it.isInfinite() }) {
                    Log.w(TAG, "Invalid sensor data detected, skipping sample")
                    return
                }

                val sample = listOf(
                    accel[2].toDouble(), accel[1].toDouble(), accel[0].toDouble(),
                    gyro[2].toDouble(), gyro[1].toDouble(), gyro[0].toDouble()
                )

                sensorData.add(sample)
                allCollectedData.add(sample)


                if (isRecordingSession) {
                    sessionData.add(sample)
                }

                collectedSamples = sensorData.size
            }

            if (collectedSamples >= TARGET_SAMPLES) {
                samplingTimer?.cancel()
                isCollecting = false
                Log.d(TAG, "Target samples reached: $collectedSamples")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in collectSample", e)
        }
    }

    private fun processCSVFiles() {
        if (selectedCombinedFile == null) {
            Toast.makeText(this, "Please select a combined CSV file with 6 columns", Toast.LENGTH_LONG).show()
            return
        }

        lifecycleScope.launch {
            try {
                isProcessingCSV = true
                csvUploadStatus = "Processing combined CSV file..."
                Log.d(TAG, "Starting combined CSV processing")

                withContext(Dispatchers.IO) {
                    val combinedData = readCombinedCSVFile(selectedCombinedFile!!)

                    if (combinedData.isEmpty()) {
                        throw Exception("Empty CSV file")
                    }

                    // Appliquer le filtrage des zéros
                    Log.d(TAG, "Applying zero filtering to CSV data...")
                    val filteredData = modelManager.preprocessCSVData(combinedData)
                    Log.d(TAG, "Zero filtering completed: ${combinedData.size} -> ${filteredData.size} samples")

                    if (filteredData.size != TARGET_SAMPLES) {
                        throw Exception("CSV file must contain exactly $TARGET_SAMPLES rows (found: ${filteredData.size})")
                    }

                    withContext(Dispatchers.Main) {
                        synchronized(sensorData) {
                            sensorData.clear()
                            sensorData.addAll(filteredData) // Utiliser les données filtrées au lieu de combinedData
                            collectedSamples = sensorData.size
                        }

                        csvUploadStatus = "Combined CSV processed successfully! $collectedSamples samples loaded"
                        Log.d(TAG, "Combined CSV processing completed: $collectedSamples samples")

                        // Arreter la capture
                        samplingTimer?.cancel()
                        isCollecting = false

                        makePredictionFromCSV()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing combined CSV file", e)
                csvUploadStatus = "Error processing CSV: ${e.message}"
                Toast.makeText(this@MainActivity, "CSV processing failed: ${e.message}", Toast.LENGTH_LONG).show()
            } finally {
                isProcessingCSV = false
            }
        }
    }
    private fun readCombinedCSVFile(uri: Uri): List<List<Double>> {
        val data = mutableListOf<List<Double>>()

        try {
            contentResolver.openInputStream(uri)?.use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    var isFirstLine = true
                    var lineNumber = 0

                    reader.forEachLine { line ->
                        lineNumber++

                        if (isFirstLine) {
                            isFirstLine = false

                            if (line.contains("acc_", ignoreCase = true) ||
                                line.contains("gyro_", ignoreCase = true) ||
                                line.contains("time", ignoreCase = true) ||
                                line.contains("sample", ignoreCase = true)) {
                                Log.d(TAG, "Combined CSV: Skipping header: $line")
                                return@forEachLine
                            }
                        }

                        try {
                            val values = line.split(",").map { it.trim() }

                            if (values.size < 6) {
                                Log.w(TAG, "Combined CSV: Line $lineNumber has only ${values.size} columns, expected 6")
                                return@forEachLine
                            }

                            // Expect columns: acc_z, acc_y, acc_x, gyro_z, gyro_y, gyro_x
                            val numericValues = values.take(6).map { it.toDouble() }

                            // Validate values
                            if (numericValues.any { it.isNaN() || it.isInfinite() }) {
                                Log.w(TAG, "Combined CSV: Invalid values in line $lineNumber: $numericValues")
                                return@forEachLine
                            }

                            data.add(numericValues)

                            // Log first few values for debugging
                            if (data.size <= 5) {
                                Log.d(TAG, "Combined CSV line $lineNumber: $numericValues")
                            }

                        } catch (e: Exception) {
                            Log.w(TAG, "Combined CSV: Error parsing line $lineNumber: $line", e)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error reading combined CSV file", e)
            throw Exception("Failed to read combined CSV: ${e.message}")
        }

        Log.d(TAG, "Combined CSV: Read ${data.size} data points")
        if (data.isNotEmpty()) {
            val allValues = data.flatten()
            Log.d(TAG, "Combined CSV data range: ${allValues.minOrNull()} to ${allValues.maxOrNull()}")
        }

        return data
    }



    private fun readCSVFile(uri: Uri, sensorType: String): List<List<Double>> {
        val data = mutableListOf<List<Double>>()

        try {
            contentResolver.openInputStream(uri)?.use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    var isFirstLine = true
                    var lineNumber = 0

                    reader.forEachLine { line ->
                        lineNumber++

                        if (isFirstLine) {
                            isFirstLine = false
                            // Skip header if it contains text
                            if (line.contains("x", ignoreCase = true) ||
                                line.contains("y", ignoreCase = true) ||
                                line.contains("z", ignoreCase = true) ||
                                line.contains("time", ignoreCase = true) ||
                                line.contains("sample", ignoreCase = true)) {
                                Log.d(TAG, "$sensorType CSV: Skipping header: $line")
                                return@forEachLine
                            }
                        }

                        try {
                            val values = line.split(",").map { it.trim() }

                            // AMÉLIORATION: Meilleure gestion des colonnes
                            val numericValues = mutableListOf<Double>()
                            for (value in values) {
                                try {
                                    val doubleValue = value.toDouble()
                                    if (!doubleValue.isNaN() && !doubleValue.isInfinite()) {
                                        numericValues.add(doubleValue)
                                    }
                                } catch (e: NumberFormatException) {
                                    // Skip non-numeric values
                                }
                            }

                            if (numericValues.size >= 3) {
                                val xyz = numericValues.take(3) // X, Y, Z


                                data.add(xyz)

                                // Log des premières valeurs pour debug
                                if (data.size <= 5) {
                                    Log.d(TAG, "$sensorType line $lineNumber: $xyz")
                                }
                            } else {
                                Log.w(TAG, "$sensorType CSV: Not enough numeric values in line $lineNumber: $numericValues")
                            }

                        } catch (e: Exception) {
                            Log.w(TAG, "$sensorType CSV: Error parsing line $lineNumber: $line", e)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error reading $sensorType CSV file", e)
            throw Exception("Failed to read $sensorType CSV: ${e.message}")
        }

        Log.d(TAG, "$sensorType CSV: Read ${data.size} data points")
        if (data.isNotEmpty()) {
            val allValues = data.flatten()
            Log.d(TAG, "$sensorType CSV data range: ${allValues.minOrNull()} to ${allValues.maxOrNull()}")
        }

        return data
    }

    private fun makePredictionFromCSV() {
        makePrediction()
    }

    private fun exportToCsv() {
        if (sensorData.isEmpty()) {
            Toast.makeText(this, "No data to export", Toast.LENGTH_SHORT).show()
            return
        }

        lifecycleScope.launch {
            try {
                isExporting = true
                Log.d(TAG, "Starting CSV export with ${sensorData.size} samples")

                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
                val fileName = "sensor_data_$timestamp.csv"

                withContext(Dispatchers.IO) {
                    // Creer fichier
                    val csvFile = File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), fileName)

                    FileWriter(csvFile).use { writer ->
                        // écrire CSV
                        writer.append("Sample,Acc_Z,Acc_Y,Acc_X,Gyro_Z,Gyro_Y,Gyro_X\n")


                        synchronized(sensorData) {
                            sensorData.forEachIndexed { index, sample ->
                                writer.append("${index + 1}")
                                sample.forEach { value ->

                                    writer.append(",${String.format(Locale.ENGLISH, "%.12f", value)}")
                                }
                                writer.append("\n")
                            }
                        }
                    }

                    Log.d(TAG, "CSV file created: ${csvFile.absolutePath}")

                    // Partager le fichier
                    withContext(Dispatchers.Main) {
                        shareFile(csvFile)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error exporting CSV", e)
                Toast.makeText(this@MainActivity, "Export failed: ${e.message}", Toast.LENGTH_LONG).show()
            } finally {
                isExporting = false
            }
        }
    }

    private fun shareFile(file: File) {
        try {
            val uri = FileProvider.getUriForFile(
                this,
                "${packageName}.fileprovider",
                file
            )

            val shareIntent = Intent().apply {
                action = Intent.ACTION_SEND
                type = "text/csv"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "Sensor Data Export")
                putExtra(Intent.EXTRA_TEXT, "Exported sensor data with ${sensorData.size} samples")
                flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
            }

            startActivity(Intent.createChooser(shareIntent, "Share CSV file"))
            Toast.makeText(this, "CSV exported successfully!", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Error sharing file", e)
            Toast.makeText(this, "File created but sharing failed. Check Downloads folder.", Toast.LENGTH_LONG).show()
        }
    }

    private fun makePrediction() {
        try {
            if (isModelLoading) {
                prediction = "Model still loading, please wait..."
                return
            }

            if (!modelManager.isInitialized) {
                prediction = "Model not initialized"
                return
            }

            if (sensorData.size < TARGET_SAMPLES) {
                prediction = "Not enough samples yet (${sensorData.size}/$TARGET_SAMPLES)"
                return
            }

            lifecycleScope.launch {
                try {
                    Log.d(TAG, "Making prediction with ${sensorData.size} samples")

                    val inputData = synchronized(sensorData) {
                        sensorData.take(TARGET_SAMPLES)
                    }

                    val processedInput = modelManager.preprocessSensorData(inputData)
                    val predictions = modelManager.runPrediction(processedInput)

                    val (predictedClass, maxProb) = predictions.maxByOrNull { it.value }
                        ?.let { it.key to it.value } ?: ("Unknown" to 0.0)

                    // Transformation des prédictions
                    // Transformation des prédictions
                    val formattedPrediction = when {
                        predictedClass.contains("brusque", ignoreCase = true) -> "Aggressive Driving"
                        predictedClass.contains("normal", ignoreCase = true) -> "Ecological  Driving"
                        else -> predictedClass
                    }

                    confidence = maxProb * 100
                    prediction = "$formattedPrediction (${String.format("%.1f", confidence)}%)"

                    Log.d(TAG, "Final prediction: $prediction")


                    if (maxProb > 0.0) {
                        savePrediction(formattedPrediction, maxProb)
                    }


                    if (selectedAccelFile == null && selectedGyroFile == null && selectedCombinedFile == null) {
                        startSampling()
                    }

                } catch (e: Exception) {
                    prediction = "Prediction error: ${e.message?.take(100)}"
                    Log.e(TAG, "Prediction error", e)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in makePrediction", e)
            prediction = "Prediction failed: ${e.message}"
        }
    }


    private fun resetCSVUpload() {
        selectedAccelFile = null
        selectedGyroFile = null
        selectedCombinedFile = null
        csvUploadStatus = ""

        // Recommencer la collecte
        if (modelManager.isInitialized) {
            startSampling()
        }

        stopAutoPrediction()
    }

    private fun startRecordingSession() {
        sessionData.clear()
        isRecordingSession = true
        startAutoPrediction()
        Log.d(TAG, "Recording session started")
    }

    private fun stopRecordingSession() {
        isRecordingSession = false
        stopAutoPrediction()
        Log.d(TAG, "Recording session stopped. Collected ${sessionData.size} samples")
    }

    private fun exportSessionData() {
        val dataToExport = if (isRecordingSession || sessionData.isNotEmpty()) {
            sessionData
        } else {
            allCollectedData
        }

        if (dataToExport.isEmpty()) {
            Toast.makeText(this, "No session data to export", Toast.LENGTH_SHORT).show()
            return
        }

        lifecycleScope.launch {
            try {
                isExporting = true
                Log.d(TAG, "Starting session CSV export with ${dataToExport.size} samples")

                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
                val fileName = "session_data_$timestamp.csv"

                withContext(Dispatchers.IO) {
                    val csvFile = File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), fileName)

                    FileWriter(csvFile).use { writer ->
                        writer.append("Sample,Acc_Z,Acc_Y,Acc_X,Gyro_Z,Gyro_Y,Gyro_X\n")

                        dataToExport.forEachIndexed { index, sample ->
                            writer.append("${index + 1}")
                            sample.forEach { value ->
                                writer.append(",${String.format(Locale.ENGLISH, "%.12f", value)}")
                            }
                            writer.append("\n")
                        }
                    }

                    Log.d(TAG, "Session CSV file created: ${csvFile.absolutePath}")

                    withContext(Dispatchers.Main) {
                        shareFile(csvFile)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error exporting session CSV", e)
                Toast.makeText(this@MainActivity, "Export failed: ${e.message}", Toast.LENGTH_LONG).show()
            } finally {
                isExporting = false
            }
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        try {
            event?.let {
                when (it.sensor.type) {
                    Sensor.TYPE_LINEAR_ACCELERATION -> {
                        lastAccelData = it.values.clone()
                        lastAccelDisplay = it.values.clone() // Mise à jour pour l'affichage
                    }
                    Sensor.TYPE_GYROSCOPE -> {
                        lastGyroData = it.values.clone()
                        lastGyroDisplay = it.values.clone() // Mise à jour pour l'affichage
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in onSensorChanged", e)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        Log.d(TAG, "Sensor accuracy changed: ${sensor?.name} -> $accuracy")
    }

    override fun onDestroy() {
        Log.d(TAG, "onDestroy() called")
        try {
            autoPredictTimer?.cancel()
            stopAutoPrediction()
            super.onDestroy()
            sensorManager.unregisterListener(this)
            samplingTimer?.cancel()
            modelManager.cleanup()
        } catch (e: Exception) {
            Log.e(TAG, "Error in onDestroy", e)
        }
    }

    override fun onPause() {
        super.onPause()
        Log.d(TAG, "onPause() - pausing sensors")
        try {
            samplingTimer?.cancel()
            isCollecting = false
        } catch (e: Exception) {
            Log.e(TAG, "Error in onPause", e)
        }
    }

    override fun onResume() {
        super.onResume()
        Log.d(TAG, "onResume() - resuming sensors")
        try {
            if (modelManager.isInitialized && !isCollecting && selectedAccelFile == null && selectedGyroFile == null) {
                startSampling()
                if (!isAutoPredicting && collectedSamples >= TARGET_SAMPLES) {
                    startAutoPrediction()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in onResume", e)
        }
    }

    @Composable
    fun MotionClassifierScreen() {
        val scrollState = rememberScrollState()

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(scrollState),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Header with current prediction
            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 6.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFFF8F9FA))
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Driving Behavior Classifier",
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF2C3E50)
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    Text(
                        text = prediction,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Medium,
                        textAlign = TextAlign.Center,
                        color = if (confidence > 70) Color(0xFF27AE60) else Color(0xFFE74C3C)
                    )

                    if (isAutoPredicting) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "🔄 AUTO PREDICTION ACTIVE",
                            fontSize = 14.sp,
                            color = Color(0xFF3498DB),
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Status
            Text(
                text = modelStatus,
                fontSize = 14.sp,
                color = if (modelManager.isInitialized) Color(0xFF27AE60) else Color(0xFFE74C3C),
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(20.dp))

            // Main buttons - Row 1
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Graph button
                Button(
                    onClick = { showGraphDialog = true },
                    enabled = allCollectedData.isNotEmpty(),
                    modifier = Modifier
                        .weight(1f)
                        .height(56.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF9C27B0)
                    )
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("\uD83D\uDD34", fontSize = 18.sp)
                        Text("Live Data", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                }

                // Auto Prediction button
                Button(
                    onClick = {
                        if (isAutoPredicting) {
                            stopRecordingSession()
                        } else {
                            startRecordingSession()
                        }
                    },
                    enabled = !isModelLoading && collectedSamples >= TARGET_SAMPLES,
                    modifier = Modifier
                        .weight(1f)
                        .height(56.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (isAutoPredicting) Color(0xFFE74C3C) else Color(0xFF27AE60)
                    )
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(if (isAutoPredicting) "⏹️" else "▶️", fontSize = 18.sp)
                        Text(
                            if (isAutoPredicting) "STOP" else "START",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Main buttons - Row 2
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Export Session button
                Button(
                    onClick = { exportSessionData() },
                    enabled = (sessionData.isNotEmpty() || allCollectedData.isNotEmpty()) && !isExporting,
                    modifier = Modifier
                        .weight(1f)
                        .height(56.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF3498DB)
                    )
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("💾", fontSize = 18.sp)
                        Text(
                            if (isExporting) "EXPORTING..." else "EXPORT",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                // CSV Upload button
                Button(
                    onClick = { pickCombinedCsvLauncher.launch("text/*") },
                    enabled = !isProcessingCSV,
                    modifier = Modifier
                        .weight(1f)
                        .height(56.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (selectedCombinedFile != null) Color(0xFF27AE60) else Color(0xFF95A5A6)
                    )
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("📁", fontSize = 18.sp)
                        Text("UPLOAD CSV", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // History button
            Button(
                onClick = { showHistoryDialog = true },
                enabled = predictionHistory.isNotEmpty(),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFFFF9800)
                )
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ) {
                    Text("📋", fontSize = 18.sp)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("HISTORY (${predictionHistory.size})", fontSize = 16.sp, fontWeight = FontWeight.Bold)
                }
            }

            Spacer(modifier = Modifier.height(20.dp))

            // CSV section if file selected
            if (selectedCombinedFile != null) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFF0F8FF))
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "📄 Selected CSV File",
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp,
                            color = Color(0xFF2C3E50)
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        Text(
                            text = getFileName(selectedCombinedFile!!),
                            fontSize = 14.sp,
                            color = Color(0xFF7F8C8D)
                        )

                        Spacer(modifier = Modifier.height(12.dp))

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            Button(
                                onClick = { processCSVFiles() },
                                enabled = !isProcessingCSV,
                                modifier = Modifier.weight(1f),
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF27AE60))
                            ) {
                                Text(
                                    if (isProcessingCSV) "PROCESSING..." else "ANALYZE",
                                    color = Color.White,
                                    fontWeight = FontWeight.Bold
                                )
                            }

                            Button(
                                onClick = { resetCSVUpload() },
                                enabled = !isProcessingCSV,
                                modifier = Modifier.weight(1f),
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFE74C3C))
                            ) {
                                Text("CANCEL", color = Color.White, fontWeight = FontWeight.Bold)
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))
            }

            // Data collection info (only when using sensors)
            if (selectedCombinedFile == null) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Data Collection",
                                fontWeight = FontWeight.Bold,
                                fontSize = 16.sp
                            )

                            Text(
                                text = if (isRecordingSession) "🔴 RECORDING" else "⚪ WAITING",
                                fontSize = 14.sp,
                                color = if (isRecordingSession) Color.Red else Color.Gray
                            )
                        }

                        Spacer(modifier = Modifier.height(12.dp))

                        LinearProgressIndicator(
                            progress = { (collectedSamples.toFloat() / TARGET_SAMPLES).coerceIn(0f, 1f) },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(8.dp),
                            color = if (collectedSamples >= TARGET_SAMPLES) Color(0xFF27AE60) else Color(0xFF3498DB)
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "Samples: $collectedSamples/$TARGET_SAMPLES",
                                fontSize = 14.sp
                            )
                            if (sessionData.isNotEmpty()) {
                                Text(
                                    text = "Session: ${sessionData.size}",
                                    fontSize = 14.sp,
                                    color = Color(0xFF3498DB)
                                )
                            }
                        }
                    }
                }
            }
        }

        // Graph dialog
        if (showGraphDialog) {
            AlertDialog(
                onDismissRequest = { showGraphDialog = false },
                title = { Text("Live Sensor Data") },
                text = {
                    Column {
                        LaunchedEffect(showGraphDialog) {
                            while (showGraphDialog) {
                                delay(10)
                            }
                        }

                        Text(
                            text = "Accelerometer (Z,Y,X):",
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "${String.format("%.3f", lastAccelDisplay[0])}, " +
                                    "${String.format("%.3f", lastAccelDisplay[1])}, " +
                                    "${String.format("%.3f", lastAccelDisplay[2])} m/s²",
                            fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        Text(
                            text = "Gyroscope (Z,Y,X):",
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "${String.format("%.3f", lastGyroDisplay[0])}, " +
                                    "${String.format("%.3f", lastGyroDisplay[1])}, " +
                                    "${String.format("%.3f", lastGyroDisplay[2])} rad/s",
                            fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        Text(
                            text = "Latest recorded values:",
                            fontWeight = FontWeight.Bold
                        )

                        LazyColumn(modifier = Modifier.height(200.dp)) {
                            items(allCollectedData.takeLast(20).reversed()) { sample ->
                                Text(
                                    text = "A:${String.format("%.3f", sample[0])},${String.format("%.3f", sample[1])},${String.format("%.3f", sample[2])} " +
                                            "G:${String.format("%.3f", sample[3])},${String.format("%.3f", sample[4])},${String.format("%.3f", sample[5])}",
                                    fontSize = 12.sp,
                                    fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                                )
                            }
                        }
                    }
                },
                confirmButton = {
                    TextButton(onClick = { showGraphDialog = false }) {
                        Text("Close")
                    }
                }
            )
        }

        // History dialog
        if (showHistoryDialog) {
            AlertDialog(
                onDismissRequest = { showHistoryDialog = false },
                title = {
                    Text("Prediction History (${predictionHistory.size})")
                },
                text = {
                    LazyColumn(
                        modifier = Modifier.height(400.dp)
                    ) {
                        items(predictionHistory) { pred ->
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 4.dp)
                            ) {
                                Text(
                                    text = "${pred.prediction}",
                                    fontWeight = FontWeight.Medium
                                )
                                Text(
                                    text = "${pred.getFormattedDate()} ${pred.getFormattedTime()} - ${pred.samples} samples",
                                    fontSize = 12.sp,
                                    color = Color.Gray
                                )
                            }
                            Divider()
                        }
                    }
                },
                confirmButton = {
                    TextButton(
                        onClick = { showHistoryDialog = false }
                    ) {
                        Text("Close")
                    }
                },
                dismissButton = {
                    TextButton(
                        onClick = {
                            clearPredictionHistory()
                            showHistoryDialog = false
                        }
                    ) {
                        Text("Clear", color = Color.Red)
                    }
                }
            )
        }
    }
    private fun startAutoPrediction() {
        if (isAutoPredicting) return

        isAutoPredicting = true
        Log.d(TAG, "Starting auto-prediction every 7 seconds")

        // Faire une première prédiction immédiate
        if (collectedSamples >= TARGET_SAMPLES && !isModelLoading) {
            makePrediction()
        }

        autoPredictTimer?.cancel()
        autoPredictTimer = Timer().apply {
            scheduleAtFixedRate(object : TimerTask() {
                override fun run() {
                    if (isAutoPredicting && collectedSamples >= TARGET_SAMPLES && !isModelLoading) {
                        makePrediction()
                    } else {
                        Log.d(TAG, "Auto-prediction skipped: collecting=$isCollecting, samples=$collectedSamples, loading=$isModelLoading")
                    }
                }
            }, 7000, 7000) // 7 seconds interval
        }
    }

    private fun stopAutoPrediction() {
        isAutoPredicting = false
        autoPredictTimer?.cancel()
        autoPredictTimer = null
        Log.d(TAG, "Auto-prediction stopped")
    }

    private fun savePrediction(predictionText: String, confidenceValue: Double) {
        lifecycleScope.launch {
            try {
                // Vérifier si la confiance est >= 70%
                if (confidenceValue < 0.7) {
                    Log.d(TAG, "Prediction confidence too low ($confidenceValue), not saving to history")
                    return@launch
                }

                // Transformation des prédictions
                val formattedPrediction = when {
                    predictionText.contains("brusque", ignoreCase = true) -> "Conduite brusque"
                    predictionText.contains("normal", ignoreCase = true) -> "Conduite normale"
                    else -> predictionText
                }

                val entity = PredictionEntity(
                    prediction = formattedPrediction,
                    confidence = confidenceValue,
                    samples = collectedSamples,
                    timestamp = System.currentTimeMillis()
                )

                Log.d(TAG, "Saving prediction: $entity")
                database.predictionDao().insertPrediction(entity)
                Log.d(TAG, "Prediction saved successfully")


                loadPredictionHistory()
            } catch (e: Exception) {
                Log.e(TAG, "Error saving prediction", e)
            }
        }
    }

    private fun loadPredictionHistory() {
        lifecycleScope.launch {
            try {
                database.predictionDao().getAllPredictions().collect { predictions ->
                    predictionHistory = predictions.map { pred ->
                        // Transformation pour l'affichage
                        pred.copy(
                            prediction = when {
                                pred.prediction.contains("brusque", ignoreCase = true) -> "Conduite brusque"
                                pred.prediction.contains("normal", ignoreCase = true) -> "Conduite normale"
                                else -> pred.prediction
                            }
                        )
                    }
                    Log.d(TAG, "Loaded ${predictions.size} predictions from database")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading prediction history", e)
            }
        }
    }

    private fun clearPredictionHistory() {
        lifecycleScope.launch {
            try {
                database.predictionDao().deleteAllPredictions()
                Toast.makeText(this@MainActivity, "History cleared", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Log.e(TAG, "Error clearing history", e)
                Toast.makeText(this@MainActivity, "Error clearing history", Toast.LENGTH_SHORT).show()
            }
        }
    }
}


class ModelManager {
    companion object {
        private const val TAG = "ModelManager"
    }

    private var interpreter: Interpreter? = null


    var labels: List<String> = emptyList()
        private set
    var isInitialized: Boolean = false
        private set

    suspend fun initialize(context: Context): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Starting ModelManager initialization with TensorFlow Lite...")

            // Changer le nom du fichier modèle
            val requiredFiles = listOf("label_encoder.json", "model_lstm.tflite")

            val assetManager = context.assets

            for (file in requiredFiles) {
                try {
                    assetManager.open(file).use { }
                    Log.d(TAG, "✓ Found required file: $file")
                } catch (e: IOException) {
                    Log.e(TAG, "✗ Missing required file: $file")
                    return@withContext false
                }
            }

            if (!loadLabels(context)) return@withContext false
            if (!initializeModel(context)) return@withContext false

            isInitialized = true
            Log.d(TAG, "ModelManager initialization completed successfully with TensorFlow Lite")
            true
        } catch (e: Exception) {
            Log.e(TAG, "ModelManager initialization error", e)
            false
        }
    }

    fun preprocessCSVData(csvData: List<List<Double>>): List<List<Double>> {
        Log.d(TAG, "Preprocessing CSV data with zero filtering...")
        return filterZerosFromData(csvData)
    }



    private fun loadLabels(context: Context): Boolean {
        return try {
            val inputStream = context.assets.open("label_encoder.json")
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            val labelData = JSONObject(jsonString)

            val classesArray = labelData.optJSONArray("classes")
            if (classesArray == null) {
                Log.e(TAG, "Invalid label format - missing classes array")
                return false
            }

            labels = (0 until classesArray.length()).map { classesArray.getString(it) }

            if (labels.isNotEmpty()) {
                Log.d(TAG, "✓ Class labels loaded successfully: $labels")
                true
            } else {
                Log.e(TAG, "No class labels found")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "✗ Failed to load class labels", e)
            false
        }
    }

    private fun initializeModel(context: Context): Boolean {
        return try {
            // Charger le modèle TensorFlow Lite
            val modelBuffer = loadModelFile(context, "model_lstm.tflite")

            // Créer l'interpréteur TensorFlow Lite
            interpreter = Interpreter(modelBuffer)

            Log.d(TAG, "✓ TensorFlow Lite Model loaded successfully")
            Log.d(TAG, "Input tensor count: ${interpreter!!.inputTensorCount}")
            Log.d(TAG, "Output tensor count: ${interpreter!!.outputTensorCount}")

            // Afficher les dimensions d'entrée et de sortie
            val inputShape = interpreter!!.getInputTensor(0).shape()
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            Log.d(TAG, "Input shape: ${inputShape.contentToString()}")
            Log.d(TAG, "Output shape: ${outputShape.contentToString()}")

            true
        } catch (e: Exception) {
            Log.e(TAG, "✗ TensorFlow Lite Model loading failed", e)
            false
        }
    }

    private fun loadModelFile(context: Context, fileName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    suspend fun runPrediction(inputData: Array<Array<FloatArray>>): Map<String, Double> = withContext(Dispatchers.Default) {
        if (!isInitialized || interpreter == null) {
            throw Exception("Model not initialized")
        }

        try {
            // Préparer les données d'entrée pour TensorFlow Lite
            // inputData a la forme [1, 700, 6] - on prend le premier élément
            val input = inputData[0] // Shape: [700, 6]

            // Créer le tensor d'entrée avec la bonne forme
            val inputArray = Array(1) { input } // Shape: [1, 700, 6]

            // Préparer le tensor de sortie
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            val outputArray = Array(outputShape[0]) { FloatArray(outputShape[1]) }

            // Exécuter l'inférence
            interpreter!!.run(inputArray, outputArray)

            // Traiter les résultats
            val output = outputArray[0]
            val predictions = mutableMapOf<String, Double>()

            for (i in labels.indices) {
                predictions[labels[i]] = output[i].toDouble()
            }

            Log.d(TAG, "TensorFlow Lite Prediction completed: ${predictions.values.maxOrNull()}")
            predictions
        } catch (e: Exception) {
            Log.e(TAG, "TensorFlow Lite Prediction error", e)
            throw Exception("Failed to run TensorFlow Lite prediction: ${e.message}")
        }
    }

    fun preprocessSensorData(rawData: List<List<Double>>): Array<Array<FloatArray>> {
        return try {
            Log.d(TAG, "Starting preprocessing with zero filtering and normalization")

            // Filtrage des zéros
            val filteredData = filterZerosFromData(rawData)

            // Normalisation MinMax à [-1, 1]
            Log.d(TAG, "Applying fixed MinMax normalization to [-1, 1]")

            // Valeurs fixes de normalisation
            // Ordre: [acc_z, acc_y, acc_x, gyro_z, gyro_y, gyro_x]
            val minValues = doubleArrayOf(-2.2413, -1.3908, -1.5099, -12.5810, -5.4858, -5.5861)
            val maxValues = doubleArrayOf(3.0627, 1.6190, 1.3043, 14.8864, 9.7007, 6.5167)

            val processedData = filteredData.map { sample ->
                sample.mapIndexed { index, value ->
                    when {
                        value.isNaN() -> {
                            Log.w(TAG, "NaN value detected at index $index, replacing with 0")
                            0f
                        }
                        value.isInfinite() -> {
                            Log.w(TAG, "Infinite value detected at index $index, clamping")
                            if (value > 0) 1f else -1f
                        }
                        else -> {
                            // MinMax normalization to [-1, 1]: 2 * (x - min) / (max - min) - 1
                            val min = minValues[index]
                            val max = maxValues[index]
                            val normalized = 2.0 * (value - min) / (max - min) - 1.0

                            normalized.coerceIn(-1.0, 1.0).toFloat()
                        }
                    }
                }.toFloatArray()
            }.toTypedArray()

            // Debug logging
            Log.d(TAG, "Raw first sample: ${rawData[0]}")
            Log.d(TAG, "Filtered first sample: ${filteredData[0]}")
            Log.d(TAG, "Normalized first sample: ${processedData[0].contentToString()}")
            val flattenedValues = processedData.flatMap { it.toList() }
            Log.d(TAG, "Final data range: ${flattenedValues.minOrNull()} to ${flattenedValues.maxOrNull()}")
            Log.d(TAG, "Final data mean: ${flattenedValues.average()}")

            arrayOf(processedData)
        } catch (e: Exception) {
            Log.e(TAG, "Preprocessing error", e)
            throw Exception("Failed to preprocess data: ${e.message}")
        }
    }

    private fun filterZerosFromData(rawData: List<List<Double>>): List<List<Double>> {
        Log.d(TAG, "Filtering zeros from sensor data...")

        // Convertir en structure plus facile à manipuler
        // rawData: List de [acc_z, acc_y, acc_x, gyro_z, gyro_y, gyro_x]
        val numColumns = 6
        val columns = Array(numColumns) { mutableListOf<Double>() }

        // Séparer les données par colonne
        rawData.forEach { sample ->
            sample.forEachIndexed { index, value ->
                if (index < numColumns) {
                    columns[index].add(value)
                }
            }
        }

        // Traiter chaque colonne individuellement
        for (colIndex in columns.indices) {
            val column = columns[colIndex]
            val columnName = when(colIndex) {
                0 -> "acc_z"
                1 -> "acc_y"
                2 -> "acc_x"
                3 -> "gyro_z"
                4 -> "gyro_y"
                5 -> "gyro_x"
                else -> "col_$colIndex"
            }

            Log.d(TAG, "Processing column $columnName: ${column.size} values")

            // Compter les zéros avant traitement
            val zeroCount = column.count { it == 0.0 }
            if (zeroCount > 0) {
                Log.d(TAG, "Found $zeroCount zeros in $columnName")

                // ÉTAPE 1: Remplacer les 0 par NaN
                val withNaN = column.map { if (it == 0.0) Double.NaN else it }.toMutableList()

                // ÉTAPE 2: Interpolation linéaire
                interpolateLinear(withNaN)

                // ÉTAPE 3: Remplacer les NaN restants par la moyenne
                val validValues = withNaN.filter { !it.isNaN() }
                if (validValues.isNotEmpty()) {
                    val mean = validValues.average()
                    for (i in withNaN.indices) {
                        if (withNaN[i].isNaN()) {
                            withNaN[i] = mean
                        }
                    }
                }

                // Remplacer la colonne originale
                columns[colIndex].clear()
                columns[colIndex].addAll(withNaN)

                Log.d(TAG, "Column $columnName processed: ${zeroCount} zeros replaced")
            }
        }

        // Reconstituer les données par échantillon
        val filteredData = mutableListOf<List<Double>>()
        for (sampleIndex in rawData.indices) {
            val sample = mutableListOf<Double>()
            for (colIndex in 0 until numColumns) {
                if (sampleIndex < columns[colIndex].size) {
                    sample.add(columns[colIndex][sampleIndex])
                }
            }
            if (sample.size == numColumns) {
                filteredData.add(sample)
            }
        }

        Log.d(TAG, "Zero filtering completed: ${rawData.size} -> ${filteredData.size} samples")
        return filteredData
    }

    private fun interpolateLinear(data: MutableList<Double>) {
        if (data.size < 2) return

        // Interpolation linéaire simple
        for (i in data.indices) {
            if (data[i].isNaN()) {
                // Trouver le point valide précédent
                var prevIndex = i - 1
                while (prevIndex >= 0 && data[prevIndex].isNaN()) {
                    prevIndex--
                }

                // Trouver le point valide suivant
                var nextIndex = i + 1
                while (nextIndex < data.size && data[nextIndex].isNaN()) {
                    nextIndex++
                }

                // Interpoler si on a des points valides des deux côtés
                if (prevIndex >= 0 && nextIndex < data.size) {
                    val prevValue = data[prevIndex]
                    val nextValue = data[nextIndex]
                    val distance = nextIndex - prevIndex
                    val position = i - prevIndex

                    // Interpolation linéaire
                    data[i] = prevValue + (nextValue - prevValue) * position / distance
                }
                // Si on n'a qu'un point précédent, utiliser sa valeur
                else if (prevIndex >= 0) {
                    data[i] = data[prevIndex]
                }
                // Si on n'a qu'un point suivant, utiliser sa valeur
                else if (nextIndex < data.size) {
                    data[i] = data[nextIndex]
                }
            }
        }
    }

    fun cleanup() {
        try {
            interpreter?.close()
            interpreter = null
            labels = emptyList()
            isInitialized = false
            Log.d(TAG, "TensorFlow Lite ModelManager cleanup completed")
        } catch (e: Exception) {
            Log.e(TAG, "Error during TensorFlow Lite cleanup", e)
        }
    }
}
// Author: Sofiane Nezar
// GitHub: https://github.com/SofianeNezar
