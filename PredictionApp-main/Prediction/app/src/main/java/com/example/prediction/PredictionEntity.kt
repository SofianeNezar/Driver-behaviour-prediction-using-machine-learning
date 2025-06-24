// Author: Sofiane Nezar
// GitHub: https://github.com/SofianeNezar

package com.example.motionclassifier

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.text.SimpleDateFormat
import java.util.*

@Entity(tableName = "predictions")
data class PredictionEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val prediction: String,
    val confidence: Double,
    val timestamp: Long = System.currentTimeMillis(),
    val samples: Int
) {
    fun getFormattedTime(): String {
        val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        return sdf.format(Date(timestamp))
    }

    fun getFormattedDate(): String {
        val sdf = SimpleDateFormat("dd/MM/yyyy", Locale.getDefault())
        return sdf.format(Date(timestamp))
    }
}
// Author: Sofiane Nezar
// GitHub: https://github.com/SofianeNezar
