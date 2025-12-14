/// Модель прогнозирования времени

use ndarray::{Array1, Array2};
use ndarray::s;
use linfa::prelude::*;
use linfa_tree::DecisionTree;
use linfa_linear::Ridge;

use crate::preprocessing::{FeatureEngineer, DataNormalizer};
use crate::types::{WeekData, ForecastingOutput};

pub struct ForecastingModel {
    tree_model: Option<DecisionTree<f64, bool>>,
    linear_model: Option<Ridge<f64>>,
    normalizer: DataNormalizer,
    is_trained: bool,
}

impl ForecastingModel {
    pub fn new() -> Self {
        Self {
            tree_model: None,
            linear_model: None,
            normalizer: DataNormalizer::new(),
            is_trained: false,
        }
    }

    pub fn train(&mut self, weeks: &[WeekData]) -> Result<(), String> {
        if weeks.len() < 8 {
            return Err("Need at least 8 weeks of data for training".to_string());
        }

        // Извлечение признаков
        let (X, y) = FeatureEngineer::extract_temporal_features(weeks)?;

        // Разделение на train/test (80/20)
        let split_idx = (X.nrows() as f64 * 0.8) as usize;
        let X_train = X.slice(s![..split_idx, ..]).to_owned();
        let X_test = X.slice(s![split_idx.., ..]).to_owned();
        let y_train = y.slice(s![..split_idx]).to_owned();
        let y_test = y.slice(s![split_idx..]).to_owned();

        // Нормализация
        let X_train_scaled = self.normalizer.fit_transform(&X_train)?;
        let X_test_scaled = self.normalizer.transform(&X_test)?;

        // Обучение Decision Tree
        let dataset = Dataset::new(X_train_scaled.clone(), y_train.clone());
        let tree_params = linfa_tree::DecisionTreeParams::new()
            .max_depth(Some(10))
            .min_samples_split(5);
        
        self.tree_model = Some(
            tree_params
                .fit(&dataset)
                .map_err(|e| format!("Tree training error: {:?}", e))?
        );

        // Обучение Linear Model (Ridge)
        let linear_params = linfa_linear::RidgeParams::new().alpha(1.0);
        self.linear_model = Some(
            linear_params
                .fit(&dataset)
                .map_err(|e| format!("Linear training error: {:?}", e))?
        );

        self.is_trained = true;

        // Оценка качества (опционально, для логирования)
        if let (Some(ref tree), Some(ref linear)) = (&self.tree_model, &self.linear_model) {
            let tree_pred = tree.predict(&Dataset::new(X_test_scaled.clone(), Array1::zeros(X_test_scaled.nrows())));
            let linear_pred = linear.predict(&Dataset::new(X_test_scaled, Array1::zeros(y_test.len())));
            
            // Ensemble
            let ensemble_pred: Array1<f64> = tree_pred.targets() * 0.7 + linear_pred.targets() * 0.3;
            
            // MAE
            let mae = (ensemble_pred - y_test).mapv(|x| x.abs()).mean().unwrap_or(0.0);
            tracing::info!("Forecasting model trained. MAE: {:.2}", mae);
        }

        Ok(())
    }

    pub fn predict(&self, weeks: &[WeekData]) -> Result<ForecastingOutput, String> {
        if !self.is_trained {
            return Err("Model not trained".to_string());
        }

        if weeks.len() < 4 {
            // Если недостаточно данных, используем среднее
            let avg_hours = if weeks.is_empty() {
                0.0
            } else {
                weeks.iter().map(|w| w.total_hours).sum::<f64>() / weeks.len() as f64
            };
            return Ok(ForecastingOutput {
                weekly_hours: avg_hours,
                weekly_hours_by_project: std::collections::HashMap::new(),
                monthly_hours: avg_hours * 4.0,
                confidence: 0.3,
                trend: "stable".to_string(),
            });
        }

        // Извлечение признаков для последней недели
        let (features, _) = FeatureEngineer::extract_temporal_features(weeks)?;
        let last_idx = features.nrows() - 1;
        let last_week_features = features.slice(ndarray::s![last_idx..last_idx+1, ..]).to_owned();

        // Нормализация
        let X_scaled = self.normalizer.transform(&last_week_features)?;

        // Предсказания
        let tree_pred = if let Some(ref tree) = self.tree_model {
            let dataset = Dataset::new(X_scaled.clone(), Array1::zeros(1));
            tree.predict(&dataset).targets()[0]
        } else {
            return Err("Tree model not available".to_string());
        };

        let linear_pred = if let Some(ref linear) = self.linear_model {
            let dataset = Dataset::new(X_scaled, Array1::zeros(1));
            linear.predict(&dataset).targets()[0]
        } else {
            return Err("Linear model not available".to_string());
        };

        // Ensemble
        let ensemble_pred = tree_pred * 0.7 + linear_pred * 0.3;

        // Confidence на основе разброса предсказаний
        let pred_std = (tree_pred - linear_pred).abs();
        let confidence = (1.0 / (1.0 + pred_std)).min(1.0);

        // Определение тренда
        let trend = if weeks.len() >= 2 {
            let recent_trend = weeks[weeks.len() - 1].total_hours - weeks[weeks.len() - 2].total_hours;
            if recent_trend > 2.0 {
                "increasing"
            } else if recent_trend < -2.0 {
                "decreasing"
            } else {
                "stable"
            }
        } else {
            "stable"
        };

        // Прогноз по проектам (упрощенный - пропорционально текущему распределению)
        let mut weekly_hours_by_project = std::collections::HashMap::new();
        if let Some(last_week) = weeks.last() {
            let total_current = last_week.total_hours;
            if total_current > 0.0 {
                for stat in &last_week.project_stats {
                    let ratio = stat.hours / total_current;
                    weekly_hours_by_project.insert(stat.project_id, ensemble_pred * ratio);
                }
            }
        }

        Ok(ForecastingOutput {
            weekly_hours: ensemble_pred,
            weekly_hours_by_project,
            monthly_hours: ensemble_pred * 4.0,
            confidence,
            trend: trend.to_string(),
        })
    }
}

impl Default for ForecastingModel {
    fn default() -> Self {
        Self::new()
    }
}

