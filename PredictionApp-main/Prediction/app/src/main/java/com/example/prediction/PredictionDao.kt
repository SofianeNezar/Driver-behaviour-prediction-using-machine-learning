// Author: Sofiane Nezar
// GitHub: https://github.com/SofianeNezar

package com.example.motionclassifier

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface PredictionDao {
    @Query("SELECT * FROM predictions ORDER BY timestamp DESC")
    fun getAllPredictions(): Flow<List<PredictionEntity>>

    @Insert
    suspend fun insertPrediction(prediction: PredictionEntity)

    @Query("DELETE FROM predictions")
    suspend fun deleteAllPredictions()
}
// Author: Sofiane Nezar
// GitHub: https://github.com/SofianeNezar
