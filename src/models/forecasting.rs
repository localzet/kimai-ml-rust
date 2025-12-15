//! Модель прогнозирования времени

#![allow(non_snake_case)]

use crate::preprocessing::{DataNormalizer, FeatureEngineer};
use crate::types::{ForecastingOutput, WeekData};
use ndarray::{s, Array1, Array2};

/// Упрощенная Ridge Regression
struct SimpleRidge {
    alpha: f64,
    weights: Option<Array1<f64>>,
    bias: Option<f64>,
}

impl SimpleRidge {
    fn new(alpha: f64) -> Self {
        Self {
            alpha,
            weights: None,
            bias: None,
        }
    }

    fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err("Empty dataset".to_string());
        }

        // Ridge Regression: (X^T X + αI)^(-1) X^T y
        // Упрощенная версия через нормальные уравнения

        // X^T X
        let mut xtx = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += X[[k, i]] * X[[k, j]];
                }
                xtx[[i, j]] = sum;
            }
        }

        // Добавляем регуляризацию (αI)
        for i in 0..n_features {
            xtx[[i, i]] += self.alpha;
        }

        // X^T y
        let mut xty = Array1::zeros(n_features);
        for i in 0..n_features {
            let mut sum = 0.0;
            for k in 0..n_samples {
                sum += X[[k, i]] * y[k];
            }
            xty[i] = sum;
        }

        // Решение через упрощенный метод (для небольших матриц)
        // В реальности нужна более сложная инверсия, но для простоты используем приближение
        self.weights = Some(self.solve_linear_system(&xtx, &xty)?);

        // Bias (среднее значение y минус среднее предсказание)
        let y_mean = y.mean().unwrap_or(0.0);
        let x_mean: Array1<f64> = (0..n_features)
            .map(|j| (0..n_samples).map(|i| X[[i, j]]).sum::<f64>() / n_samples as f64)
            .collect();

        if let Some(ref weights) = self.weights {
            let pred_mean: f64 = x_mean.iter().zip(weights.iter()).map(|(x, w)| x * w).sum();
            self.bias = Some(y_mean - pred_mean);
        }

        Ok(())
    }

    fn solve_linear_system(&self, A: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
        // Упрощенное решение через метод Гаусса (для небольших систем)
        let n = A.nrows();
        let mut augmented = Array2::zeros((n, n + 1));

        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = A[[i, j]];
            }
            augmented[[i, n]] = b[i];
        }

        // Прямой ход метода Гаусса
        for i in 0..n {
            // Поиск максимального элемента в столбце
            let mut max_row = i;
            let mut max_val = augmented[[i, i]].abs();
            for k in (i + 1)..n {
                if augmented[[k, i]].abs() > max_val {
                    max_val = augmented[[k, i]].abs();
                    max_row = k;
                }
            }

            // Перестановка строк
            if max_row != i {
                for j in 0..=n {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Исключение
            let pivot = augmented[[i, i]];
            if pivot.abs() < 1e-10 {
                return Err("Singular matrix".to_string());
            }

            for k in (i + 1)..n {
                let factor = augmented[[k, i]] / pivot;
                for j in i..=n {
                    augmented[[k, j]] -= factor * augmented[[i, j]];
                }
            }
        }

        // Обратный ход
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = augmented[[i, n]];
            for j in (i + 1)..n {
                sum -= augmented[[i, j]] * x[j];
            }
            x[i] = sum / augmented[[i, i]];
        }

        Ok(x)
    }

    fn predict(&self, X: &Array2<f64>) -> Result<Array1<f64>, String> {
        let weights = self.weights.as_ref().ok_or("Model not trained")?;
        let bias = self.bias.unwrap_or(0.0);

        let mut predictions = Array1::zeros(X.nrows());
        for i in 0..X.nrows() {
            let mut pred = bias;
            for j in 0..X.ncols() {
                pred += X[[i, j]] * weights[j];
            }
            predictions[i] = pred;
        }

        Ok(predictions)
    }
}

/// Упрощенный Decision Tree (регрессия)
struct SimpleTree {
    max_depth: usize,
    min_samples_split: usize,
    root: Option<TreeNode>,
}

enum TreeNode {
    Leaf {
        value: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl SimpleTree {
    fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            max_depth,
            min_samples_split,
            root: None,
        }
    }

    fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        if X.nrows() == 0 {
            return Err("Empty dataset".to_string());
        }

        self.root = Some(self.build_tree(X, y, 0, (0..X.nrows()).collect()));
        Ok(())
    }

    fn build_tree(
        &self,
        X: &Array2<f64>,
        y: &Array1<f64>,
        depth: usize,
        indices: Vec<usize>,
    ) -> TreeNode {
        if depth >= self.max_depth || indices.len() < self.min_samples_split {
            // Лист: среднее значение
            let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len() as f64;
            return TreeNode::Leaf { value: mean };
        }

        // Поиск лучшего разделения
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_score = f64::INFINITY;

        for feature in 0..X.ncols() {
            let values: Vec<f64> = indices.iter().map(|&i| X[[i, feature]]).collect();
            let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            if (max_val - min_val).abs() < 1e-10 {
                continue;
            }

            // Пробуем несколько порогов
            for _ in 0..10 {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let threshold = rng.gen_range(min_val..=max_val);

                let (left_indices, right_indices): (Vec<usize>, Vec<usize>) =
                    indices.iter().partition(|&&i| X[[i, feature]] < threshold);

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                // Вычисляем MSE
                let left_mean =
                    left_indices.iter().map(|&i| y[i]).sum::<f64>() / left_indices.len() as f64;
                let right_mean =
                    right_indices.iter().map(|&i| y[i]).sum::<f64>() / right_indices.len() as f64;

                let left_mse: f64 = left_indices
                    .iter()
                    .map(|&i| (y[i] - left_mean).powi(2))
                    .sum();
                let right_mse: f64 = right_indices
                    .iter()
                    .map(|&i| (y[i] - right_mean).powi(2))
                    .sum();
                let total_mse = left_mse + right_mse;

                if total_mse < best_score {
                    best_score = total_mse;
                    best_feature = feature;
                    best_threshold = threshold;
                }
            }
        }

        if best_score == f64::INFINITY {
            // Не удалось найти хорошее разделение
            let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len() as f64;
            return TreeNode::Leaf { value: mean };
        }

        // Разделение
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| X[[i, best_feature]] < best_threshold);

        TreeNode::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(self.build_tree(X, y, depth + 1, left_indices)),
            right: Box::new(self.build_tree(X, y, depth + 1, right_indices)),
        }
    }

    fn predict(&self, X: &Array2<f64>) -> Result<Array1<f64>, String> {
        let root = self.root.as_ref().ok_or("Model not trained")?;
        let mut predictions = Array1::zeros(X.nrows());

        for i in 0..X.nrows() {
            predictions[i] = self.predict_single(root, &X.row(i).to_owned());
        }

        Ok(predictions)
    }

    fn predict_single(&self, node: &TreeNode, sample: &Array1<f64>) -> f64 {
        match node {
            TreeNode::Leaf { value } => *value,
            TreeNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if sample[*feature] < *threshold {
                    self.predict_single(left, sample)
                } else {
                    self.predict_single(right, sample)
                }
            }
        }
    }
}

pub struct ForecastingModel {
    tree_model: Option<SimpleTree>,
    linear_model: Option<SimpleRidge>,
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
        let mut tree = SimpleTree::new(10, 5);
        tree.fit(&X_train_scaled, &y_train)?;
        self.tree_model = Some(tree);

        // Обучение Linear Model (Ridge)
        let mut linear = SimpleRidge::new(1.0);
        linear.fit(&X_train_scaled, &y_train)?;
        self.linear_model = Some(linear);

        self.is_trained = true;

        // Оценка качества (опционально, для логирования)
        if let (Some(ref tree), Some(ref linear)) = (&self.tree_model, &self.linear_model) {
            let tree_pred = tree.predict(&X_test_scaled)?;
            let linear_pred = linear.predict(&X_test_scaled)?;

            // Ensemble
            let ensemble_pred: Array1<f64> = tree_pred * 0.7 + linear_pred * 0.3;

            // MAE
            let mae = (ensemble_pred - y_test)
                .mapv(|x| x.abs())
                .mean()
                .unwrap_or(0.0);
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
        let last_week_features = features.slice(s![last_idx..last_idx + 1, ..]).to_owned();

        // Нормализация
        let X_scaled = self.normalizer.transform(&last_week_features)?;

        // Предсказания
        let tree_pred = if let Some(ref tree) = self.tree_model {
            let pred = tree.predict(&X_scaled)?;
            pred[0]
        } else {
            return Err("Tree model not available".to_string());
        };

        let linear_pred = if let Some(ref linear) = self.linear_model {
            let pred = linear.predict(&X_scaled)?;
            pred[0]
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
            let recent_trend =
                weeks[weeks.len() - 1].total_hours - weeks[weeks.len() - 2].total_hours;
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

        // Прогноз по проектам с учетом целей пользователя
        let mut weekly_hours_by_project = std::collections::HashMap::new();
        if let Some(last_week) = weeks.last() {
            let total_current = last_week.total_hours;

            if total_current > 0.0 {
                // Без целей - пропорционально текущему распределению
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
