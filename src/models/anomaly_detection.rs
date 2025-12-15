//! Обнаружение аномалий в записях времени

use ndarray::Array2;

use crate::preprocessing::FeatureEngineer;
use crate::types::{TimesheetEntry, AnomalyOutput};

/// Упрощенный Isolation Forest
pub struct IsolationForest {
    n_trees: usize,
    max_samples: usize,
    max_depth: usize,
    trees: Vec<IsolationTree>,
}

enum IsolationTree {
    Leaf,
    Split {
        feature: usize,
        threshold: f64,
        left: Box<IsolationTree>,
        right: Box<IsolationTree>,
    },
}

impl IsolationForest {
    pub fn new(n_trees: usize, max_samples: usize, max_depth: usize) -> Self {
        Self {
            n_trees,
            max_samples,
            max_depth,
            trees: Vec::new(),
        }
    }

    pub fn fit(&mut self, features: &Array2<f64>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..self.n_trees {
            // Случайная выборка
            let mut indices: Vec<usize> = (0..features.nrows()).collect();
            for _ in 0..(features.nrows().saturating_sub(self.max_samples)) {
                if !indices.is_empty() {
                    let idx = rng.gen_range(0..indices.len());
                    indices.remove(idx);
                }
            }

            // Построение дерева
            let tree = self.build_tree(features, &indices, 0);
            self.trees.push(IsolationTree::Split {
                feature: 0,
                threshold: 0.0,
                left: Box::new(tree),
                right: Box::new(IsolationTree::Leaf),
            });
        }
    }

    fn build_tree(&self, features: &Array2<f64>, indices: &[usize], depth: usize) -> IsolationTree {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        if depth >= self.max_depth || indices.len() <= 1 {
            return IsolationTree::Leaf;
        }

        let feature = rng.gen_range(0..features.ncols());

        // Случайный порог
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &idx in indices {
            let val = features[[idx, feature]];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
        let threshold = rng.gen_range(min_val..=max_val);

        // Разделение
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| features[[i, feature]] < threshold);

        if left_indices.is_empty() || right_indices.is_empty() {
            return IsolationTree::Leaf;
        }

        IsolationTree::Split {
            feature,
            threshold,
            left: Box::new(self.build_tree(features, &left_indices, depth + 1)),
            right: Box::new(self.build_tree(features, &right_indices, depth + 1)),
        }
    }

    pub fn predict(&self, features: &Array2<f64>) -> Vec<f64> {
        let mut scores = vec![0.0; features.nrows()];

        for tree in &self.trees {
            for (i, row) in features.rows().into_iter().enumerate() {
                let path_length = self.path_length(tree, &row.to_owned(), 0);
                scores[i] += path_length;
            }
        }

        // Нормализация
        let n_trees = self.n_trees as f64;
        for score in &mut scores {
            *score /= n_trees;
        }

        // Преобразование в anomaly score (чем короче путь, тем выше аномальность)
        scores.iter().map(|s| (-s).exp()).collect()
    }

    fn path_length(&self, node: &IsolationTree, sample: &ndarray::Array1<f64>, current_depth: usize) -> f64 {
        match node {
            IsolationTree::Leaf => current_depth as f64,
            IsolationTree::Split { feature, threshold, left, right } => {
                if sample[*feature] < *threshold {
                    self.path_length(left, sample, current_depth + 1)
                } else {
                    self.path_length(right, sample, current_depth + 1)
                }
            }
        }
    }
}

pub struct AnomalyDetector {
    isolation_forest: Option<IsolationForest>,
    contamination: f64,
    is_trained: bool,
}

impl AnomalyDetector {
    pub fn new(contamination: f64) -> Self {
        Self {
            isolation_forest: None,
            contamination,
            is_trained: false,
        }
    }

    pub fn train(&mut self, entries: &[TimesheetEntry]) -> Result<(), String> {
        if entries.len() < 20 {
            return Err("Need at least 20 entries for training".to_string());
        }

        let features = FeatureEngineer::extract_anomaly_features(entries);
        
        let max_samples = (entries.len() as f64 * 0.8) as usize;
        let mut forest = IsolationForest::new(100, max_samples, 10);
        forest.fit(&features);

        self.isolation_forest = Some(forest);
        self.is_trained = true;

        Ok(())
    }

    pub fn detect(&self, entries: &[TimesheetEntry]) -> Result<Vec<AnomalyOutput>, String> {
        if !self.is_trained {
            return Err("Detector not trained".to_string());
        }

        if entries.is_empty() {
            return Ok(Vec::new());
        }

        let features = FeatureEngineer::extract_anomaly_features(entries);
        let forest = self.isolation_forest.as_ref().ok_or("Forest not available")?;
        
        let scores = forest.predict(&features);

        // Нормализация scores к [0, 1]
        let min_score = scores.iter().copied().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let score_range = (max_score - min_score).max(1e-10);
        
        let normalized_scores: Vec<f64> = scores
            .iter()
            .map(|s| 1.0 - (s - min_score) / score_range)
            .collect();

        let mut anomalies = Vec::new();

        for (i, entry) in entries.iter().enumerate() {
            let score = normalized_scores[i];
            
            // Порог для аномалии (на основе contamination)
            if score > self.contamination {
                let severity = self.determine_severity(entry, score);
                let anomaly_type = self.classify_anomaly_type(entry);
                let reason = self.generate_reason(entry, score);

                anomalies.push(AnomalyOutput {
                    entry_id: entry.id,
                    r#type: anomaly_type,
                    severity,
                    reason,
                    score,
                });
            }
        }

        Ok(anomalies)
    }

    fn determine_severity(&self, entry: &TimesheetEntry, score: f64) -> String {
        let mut severity_score = score;

        if entry.duration > 10 * 60 {
            severity_score += 0.2;
        } else if entry.duration < 5 {
            severity_score += 0.1;
        }

        if entry.hour_of_day < 5 || entry.hour_of_day > 23 {
            severity_score += 0.15;
        }

        if severity_score > 0.8 {
            "high".to_string()
        } else if severity_score > 0.5 {
            "medium".to_string()
        } else {
            "low".to_string()
        }
    }

    fn classify_anomaly_type(&self, entry: &TimesheetEntry) -> String {
        if entry.duration > 8 * 60 || entry.duration < 5 {
            "duration".to_string()
        } else if entry.hour_of_day < 6 || entry.hour_of_day > 23 {
            "time".to_string()
        } else {
            "pattern".to_string()
        }
    }

    fn generate_reason(&self, entry: &TimesheetEntry, score: f64) -> String {
        let mut reasons = Vec::new();

        if entry.duration > 8 * 60 {
            reasons.push(format!("Очень длинная сессия: {:.1} часов", entry.duration as f64 / 60.0));
        } else if entry.duration < 5 {
            reasons.push(format!("Очень короткая сессия: {} минут", entry.duration));
        }

        if entry.hour_of_day < 6 {
            reasons.push(format!("Работа в ночное время: {}:00", entry.hour_of_day));
        } else if entry.hour_of_day > 23 {
            reasons.push(format!("Работа поздно вечером: {}:00", entry.hour_of_day));
        }

        if score > 0.7 {
            reasons.push("Необычный паттерн работы".to_string());
        }

        if reasons.is_empty() {
            "Обнаружена аномалия".to_string()
        } else {
            reasons.join("; ")
        }
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new(0.1)
    }
}

