//! Kimai ML - Rust библиотека

pub mod types;
pub mod models;
pub mod preprocessing;

pub use types::*;
pub use models::*;
pub use preprocessing::*;

// Re-export для удобства
pub use models::learning::{LearningModule, PredictionError};

