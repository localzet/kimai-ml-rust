/// Обучение на ошибках - улучшение моделей на основе фактических результатов

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionError {
    pub prediction_type: String,
    pub predicted_value: f64,
    pub actual_value: f64,
    pub error: f64,
    pub context: serde_json::Value,
}

pub struct LearningModule {
    errors: Vec<PredictionError>,
    max_errors: usize,
}

impl LearningModule {
    pub fn new(max_errors: usize) -> Self {
        Self {
            errors: Vec::new(),
            max_errors,
        }
    }

    pub fn record_error(&mut self, error: PredictionError) {
        self.errors.push(error);
        if self.errors.len() > self.max_errors {
            self.errors.remove(0);
        }
    }

    pub fn get_correction_factor(&self, prediction_type: &str) -> f64 {
        let relevant_errors: Vec<&PredictionError> = self.errors
            .iter()
            .filter(|e| e.prediction_type == prediction_type)
            .collect();

        if relevant_errors.is_empty() {
            return 1.0;
        }

        // Вычисляем среднюю ошибку
        let avg_error: f64 = relevant_errors.iter()
            .map(|e| e.error.abs())
            .sum::<f64>() / relevant_errors.len() as f64;

        // Вычисляем средний процент ошибки
        let avg_percent_error: f64 = relevant_errors.iter()
            .filter(|e| e.actual_value != 0.0)
            .map(|e| (e.error / e.actual_value).abs())
            .sum::<f64>() / relevant_errors.iter().filter(|e| e.actual_value != 0.0).count() as f64;

        // Корректирующий фактор: если предсказания завышены, уменьшаем, если занижены - увеличиваем
        let bias: f64 = relevant_errors.iter()
            .map(|e| e.error)
            .sum::<f64>() / relevant_errors.len() as f64;

        // Если есть систематическая ошибка (bias), корректируем
        if bias.abs() > avg_error * 0.1 {
            // Корректируем на основе bias
            1.0 - (bias / avg_error).signum() * avg_percent_error.min(0.2)
        } else {
            1.0
        }
    }

    pub fn get_confidence_adjustment(&self, prediction_type: &str) -> f64 {
        let relevant_errors: Vec<&PredictionError> = self.errors
            .iter()
            .filter(|e| e.prediction_type == prediction_type)
            .collect();

        if relevant_errors.is_empty() {
            return 1.0;
        }

        // Вычисляем стандартное отклонение ошибок
        let avg_error: f64 = relevant_errors.iter()
            .map(|e| e.error.abs())
            .sum::<f64>() / relevant_errors.len() as f64;

        let variance: f64 = relevant_errors.iter()
            .map(|e| {
                let diff = e.error.abs() - avg_error;
                diff * diff
            })
            .sum::<f64>() / relevant_errors.len() as f64;

        let std_dev = variance.sqrt();

        // Если ошибки стабильны (низкое std_dev), увеличиваем уверенность
        // Если ошибки нестабильны (высокое std_dev), уменьшаем уверенность
        if avg_error > 0.0 {
            let coefficient_of_variation = std_dev / avg_error;
            // Нормализуем к диапазону [0.5, 1.0]
            (1.0 / (1.0 + coefficient_of_variation)).max(0.5).min(1.0)
        } else {
            1.0
        }
    }

    pub fn analyze_patterns(&self) -> HashMap<String, f64> {
        let mut patterns = HashMap::new();

        // Анализ ошибок по типам
        let mut errors_by_type: HashMap<String, Vec<f64>> = HashMap::new();
        for error in &self.errors {
            errors_by_type
                .entry(error.prediction_type.clone())
                .or_insert_with(Vec::new)
                .push(error.error.abs());
        }

        for (pred_type, errors) in errors_by_type {
            if !errors.is_empty() {
                let avg_error = errors.iter().sum::<f64>() / errors.len() as f64;
                patterns.insert(pred_type, avg_error);
            }
        }

        patterns
    }
}

impl Default for LearningModule {
    fn default() -> Self {
        Self::new(1000)
    }
}

