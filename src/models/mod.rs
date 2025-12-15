//! ML модели

pub mod anomaly_detection;
pub mod forecasting;
pub mod learning;
pub mod productivity;
pub mod recommendations;

pub use anomaly_detection::AnomalyDetector;
pub use forecasting::ForecastingModel;
pub use learning::{LearningModule, PredictionError};
pub use productivity::ProductivityAnalyzer;
pub use recommendations::RecommendationEngine;
