/// API сервер для ML моделей

use axum::{
    extract::State,
    http::Method,
    response::Json,
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tracing_subscriber;

use kimai_ml::{
    types::{MLInputData, MLOutputData},
    ForecastingModel, AnomalyDetector, RecommendationEngine, ProductivityAnalyzer,
};

#[derive(Clone)]
struct AppState {
    forecasting_model: std::sync::Arc<tokio::sync::Mutex<ForecastingModel>>,
    anomaly_detector: std::sync::Arc<tokio::sync::Mutex<AnomalyDetector>>,
    recommendation_engine: std::sync::Arc<tokio::sync::Mutex<RecommendationEngine>>,
    productivity_analyzer: std::sync::Arc<tokio::sync::Mutex<ProductivityAnalyzer>>,
}

#[tokio::main]
async fn main() {
    // Инициализация логирования
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let state = AppState {
        forecasting_model: std::sync::Arc::new(tokio::sync::Mutex::new(ForecastingModel::new())),
        anomaly_detector: std::sync::Arc::new(tokio::sync::Mutex::new(AnomalyDetector::new(0.1))),
        recommendation_engine: std::sync::Arc::new(tokio::sync::Mutex::new(RecommendationEngine::new())),
        productivity_analyzer: std::sync::Arc::new(tokio::sync::Mutex::new(ProductivityAnalyzer)),
    };

    // CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/api/predict", post(predict))
        .route("/api/detect-anomalies", post(detect_anomalies))
        .route("/api/recommendations", post(get_recommendations))
        .route("/api/productivity", post(analyze_productivity))
        .layer(cors)
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], 8000));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    tracing::info!("Server listening on http://0.0.0.0:8000");
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "message": "Kimai ML API (Rust)",
        "version": "0.1.0"
    }))
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn predict(
    State(state): State<AppState>,
    Json(data): Json<MLInputData>,
) -> Result<Json<MLOutputData>, String> {
    tracing::info!("Predict request: {} weeks, {} entries", data.weeks.len(), data.timesheets.len());

    let weeks: Vec<kimai_ml::types::WeekData> = data.weeks.iter().map(|w| kimai_ml::types::WeekData {
        year: w.year,
        week: w.week,
        total_minutes: w.total_minutes,
        total_hours: w.total_hours,
        total_amount: w.total_amount,
        project_stats: w.project_stats.iter().map(|s| kimai_ml::types::ProjectStats {
            project_id: s.project_id,
            minutes: s.minutes,
            hours: s.hours,
        }).collect(),
    }).collect();

    let mut model = state.forecasting_model.lock().await;

    if weeks.len() < 8 {
        let avg_hours = if weeks.is_empty() {
            0.0
        } else {
            weeks.iter().map(|w| w.total_hours).sum::<f64>() / weeks.len() as f64
        };
        return Ok(Json(MLOutputData {
            forecasting: Some(kimai_ml::types::ForecastingOutput {
                weekly_hours: avg_hours,
                weekly_hours_by_project: std::collections::HashMap::new(),
                monthly_hours: avg_hours * 4.0,
                confidence: 0.3,
                trend: "stable".to_string(),
            }),
            anomalies: None,
            recommendations: None,
            productivity: None,
        }));
    }

    // Обучение (если еще не обучена)
    if let Err(e) = model.train(&weeks) {
        tracing::warn!("Training failed: {}", e);
    }

    // Прогнозирование
    match model.predict(&weeks) {
        Ok(forecasting) => Ok(Json(MLOutputData {
            forecasting: Some(forecasting),
            anomalies: None,
            recommendations: None,
            productivity: None,
        })),
        Err(e) => Err(format!("Prediction error: {}", e)),
    }
}

async fn detect_anomalies(
    State(state): State<AppState>,
    Json(data): Json<MLInputData>,
) -> Result<Json<MLOutputData>, String> {
    tracing::info!("Detect anomalies request: {} entries", data.timesheets.len());

    if data.timesheets.is_empty() {
        return Ok(Json(MLOutputData {
            forecasting: None,
            anomalies: Some(Vec::new()),
            recommendations: None,
            productivity: None,
        }));
    }

    let entries: Vec<kimai_ml::types::TimesheetEntry> = data.timesheets.iter().map(|e| kimai_ml::types::TimesheetEntry {
        id: e.id,
        begin: e.begin.clone(),
        end: e.end.clone(),
        duration: e.duration,
        project_id: e.project_id,
        project_name: e.project_name.clone(),
        activity_id: e.activity_id,
        activity_name: e.activity_name.clone(),
        description: e.description.clone(),
        tags: e.tags.clone(),
        day_of_week: e.day_of_week,
        hour_of_day: e.hour_of_day,
        week_of_year: e.week_of_year,
        month: e.month,
        year: e.year,
    }).collect();

    let mut detector = state.anomaly_detector.lock().await;

    if entries.len() >= 20 {
        if let Err(e) = detector.train(&entries) {
            tracing::warn!("Training failed: {}", e);
        }
    }

    match detector.detect(&entries) {
        Ok(anomalies) => Ok(Json(MLOutputData {
            forecasting: None,
            anomalies: Some(anomalies),
            recommendations: None,
            productivity: None,
        })),
        Err(e) => Err(format!("Detection error: {}", e)),
    }
}

async fn get_recommendations(
    State(state): State<AppState>,
    Json(data): Json<MLInputData>,
) -> Result<Json<MLOutputData>, String> {
    tracing::info!("Recommendations request: {} projects", data.projects.len());

    let mut engine = state.recommendation_engine.lock().await;
    let recommendations = engine.generate_recommendations(&data);

    Ok(Json(MLOutputData {
        forecasting: None,
        anomalies: None,
        recommendations: Some(recommendations),
        productivity: None,
    }))
}

async fn analyze_productivity(
    State(state): State<AppState>,
    Json(data): Json<MLInputData>,
) -> Result<Json<MLOutputData>, String> {
    tracing::info!("Productivity analysis request: {} entries", data.timesheets.len());

    if data.timesheets.is_empty() {
        return Err("No timesheet entries provided".to_string());
    }

    let entries: Vec<kimai_ml::types::TimesheetEntry> = data.timesheets.iter().map(|e| kimai_ml::types::TimesheetEntry {
        id: e.id,
        begin: e.begin.clone(),
        end: e.end.clone(),
        duration: e.duration,
        project_id: e.project_id,
        project_name: e.project_name.clone(),
        activity_id: e.activity_id,
        activity_name: e.activity_name.clone(),
        description: e.description.clone(),
        tags: e.tags.clone(),
        day_of_week: e.day_of_week,
        hour_of_day: e.hour_of_day,
        week_of_year: e.week_of_year,
        month: e.month,
        year: e.year,
    }).collect();

    let analyzer = state.productivity_analyzer.lock().await;
    let productivity = analyzer.analyze(&entries);

    Ok(Json(MLOutputData {
        forecasting: None,
        anomalies: None,
        recommendations: None,
        productivity: Some(productivity),
    }))
}

