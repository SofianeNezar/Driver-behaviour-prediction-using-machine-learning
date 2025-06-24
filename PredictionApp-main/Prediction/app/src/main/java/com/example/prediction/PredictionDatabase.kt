// Author: Sofiane Nezar
// GitHub: https://github.com/SofianeNezar

package com.example.motionclassifier

import android.content.Context
import android.util.Log
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [PredictionEntity::class],
    version = 1,
    exportSchema = false
)
abstract class PredictionDatabase : RoomDatabase() {
    abstract fun predictionDao(): PredictionDao

    companion object {
        private const val TAG = "PredictionDatabase"

        @Volatile
        private var INSTANCE: PredictionDatabase? = null

        fun getDatabase(context: Context): PredictionDatabase {
            return INSTANCE ?: synchronized(this) {
                Log.d(TAG, "Creating new database instance")
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    PredictionDatabase::class.java,
                    "prediction_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}
