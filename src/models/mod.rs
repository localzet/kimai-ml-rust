/// ML модели

pub mod forecasting;
pub mod anomaly_detection;
pub mod recommendations;
pub mod productivity;

pub use forecasting::ForecastingModel;
pub use anomaly_detection::AnomalyDetector;
pub use recommendations::RecommendationEngine;
pub use productivity::ProductivityAnalyzer;

