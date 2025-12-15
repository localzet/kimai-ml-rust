//! Нормализация данных

#![allow(non_snake_case)]

use ndarray::{Array1, Array2, Axis};

pub struct DataNormalizer {
    mean: Option<Array1<f64>>,
    std: Option<Array1<f64>>,
    is_fitted: bool,
}

impl DataNormalizer {
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
            is_fitted: false,
        }
    }

    pub fn fit(&mut self, X: &Array2<f64>) -> Result<(), String> {
        if X.nrows() == 0 {
            return Err("Empty dataset".to_string());
        }

        // Вычисляем среднее и стандартное отклонение по каждому признаку
        self.mean = Some(X.mean_axis(Axis(0)).ok_or("Failed to compute mean")?);
        self.std = Some(X.std_axis(Axis(0), 0.0));

        // Избегаем деления на ноль
        if let Some(ref mut std) = self.std {
            for val in std.iter_mut() {
                if *val < 1e-10 {
                    *val = 1.0;
                }
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    pub fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>, String> {
        if !self.is_fitted {
            return Err("Normalizer not fitted".to_string());
        }

        let mean = self.mean.as_ref().ok_or("Mean not computed")?;
        let std = self.std.as_ref().ok_or("Std not computed")?;

        // Нормализация: (X - mean) / std
        let mut normalized = X.clone();
        for mut row in normalized.rows_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                *val = (*val - mean[i]) / std[i];
            }
        }

        Ok(normalized)
    }

    pub fn fit_transform(&mut self, X: &Array2<f64>) -> Result<Array2<f64>, String> {
        self.fit(X)?;
        self.transform(X)
    }
}

impl Default for DataNormalizer {
    fn default() -> Self {
        Self::new()
    }
}
