//! Feature engineering для ML моделей

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

use crate::types::{TimesheetEntry, WeekData};

pub struct FeatureEngineer;

impl FeatureEngineer {
    //! Извлечение временных признаков из недель
    pub fn extract_temporal_features(
        weeks: &[WeekData],
    ) -> Result<(Array2<f64>, Array1<f64>), String> {
        if weeks.is_empty() {
            return Err("No weeks provided".to_string());
        }

        let n_samples = weeks.len();
        let n_features = 13; // Количество признаков

        let mut features = Array2::zeros((n_samples, n_features));
        let mut targets = Array1::zeros(n_samples);

        for (i, week) in weeks.iter().enumerate() {
            let mut feature_idx = 0;

            // Базовые временные признаки
            features[[i, feature_idx]] = week.week as f64;
            feature_idx += 1;
            features[[i, feature_idx]] = week.year as f64;
            feature_idx += 1;

            // Месяц (приблизительно из недели)
            let month = ((week.week - 1) / 4) + 1;
            features[[i, feature_idx]] = month as f64;
            feature_idx += 1;

            // Циклические признаки
            features[[i, feature_idx]] = (2.0 * PI * week.week as f64 / 52.0).sin();
            feature_idx += 1;
            features[[i, feature_idx]] = (2.0 * PI * week.week as f64 / 52.0).cos();
            feature_idx += 1;
            features[[i, feature_idx]] = (2.0 * PI * month as f64 / 12.0).sin();
            feature_idx += 1;
            features[[i, feature_idx]] = (2.0 * PI * month as f64 / 12.0).cos();
            feature_idx += 1;

            // Исторические признаки
            if i > 0 {
                features[[i, feature_idx]] = weeks[i - 1].total_hours;
            }
            feature_idx += 1;

            if i >= 4 {
                let avg: f64 = weeks[i - 4..i].iter().map(|w| w.total_hours).sum::<f64>() / 4.0;
                features[[i, feature_idx]] = avg;
            }
            feature_idx += 1;

            if i >= 8 {
                let avg: f64 = weeks[i - 8..i].iter().map(|w| w.total_hours).sum::<f64>() / 8.0;
                features[[i, feature_idx]] = avg;
            }
            feature_idx += 1;

            // Тренд (упрощенный)
            if i >= 4 {
                let recent: Vec<f64> = weeks[i - 4..i].iter().map(|w| w.total_hours).collect();
                if recent.len() >= 2 {
                    let trend = (recent[recent.len() - 1] - recent[0]) / recent.len() as f64;
                    features[[i, feature_idx]] = trend;
                }
            }
            feature_idx += 1;

            // Волатильность
            if i >= 4 {
                let values: Vec<f64> = weeks[i - 4..i].iter().map(|w| w.total_hours).collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                features[[i, feature_idx]] = variance.sqrt();
            }

            // Целевая переменная
            targets[i] = week.total_hours;
        }

        Ok((features, targets))
    }

    /// Извлечение признаков для обнаружения аномалий
    pub fn extract_anomaly_features(entries: &[TimesheetEntry]) -> Array2<f64> {
        if entries.is_empty() {
            return Array2::zeros((0, 5));
        }

        // Вычисляем среднюю длительность по проектам
        use std::collections::HashMap;
        let mut project_durations: HashMap<i32, Vec<i32>> = HashMap::new();
        for entry in entries {
            if let Some(project_id) = entry.project_id {
                project_durations
                    .entry(project_id)
                    .or_default()
                    .push(entry.duration);
            }
        }

        let mut project_avg: HashMap<i32, f64> = HashMap::new();
        for (project_id, durations) in project_durations {
            let avg = durations.iter().sum::<i32>() as f64 / durations.len() as f64;
            project_avg.insert(project_id, avg);
        }

        let n_samples = entries.len();
        let n_features = 5;
        let mut features = Array2::zeros((n_samples, n_features));

        for (i, entry) in entries.iter().enumerate() {
            // Нормализованная длительность (0-1, нормализация к 8 часам)
            let duration_norm = (entry.duration as f64 / (8.0 * 60.0)).min(1.0);
            features[[i, 0]] = duration_norm;

            // Время дня (0-1)
            features[[i, 1]] = entry.hour_of_day as f64 / 23.0;

            // День недели (0-1)
            features[[i, 2]] = entry.day_of_week as f64 / 6.0;

            // Отношение к среднему по проекту
            let project_avg_val = entry
                .project_id
                .and_then(|id| project_avg.get(&id))
                .copied()
                .unwrap_or(entry.duration as f64);
            let duration_ratio = if project_avg_val > 0.0 {
                (entry.duration as f64 / project_avg_val).min(5.0)
            } else {
                1.0
            };
            features[[i, 3]] = duration_ratio;

            // Количество тегов
            features[[i, 4]] = entry.tags.len() as f64;
        }

        features
    }
}
