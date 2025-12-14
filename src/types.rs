/// Типы данных для ML модуля

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimesheetEntry {
    pub id: i32,
    pub begin: String,
    pub end: Option<String>,
    pub duration: i32, // минуты
    pub project_id: Option<i32>,
    pub project_name: String,
    pub activity_id: Option<i32>,
    pub activity_name: String,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub day_of_week: i32,
    pub hour_of_day: i32,
    pub week_of_year: i32,
    pub month: i32,
    pub year: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: i32,
    pub name: String,
    pub total_hours: f64,
    pub avg_hours_per_week: f64,
    pub weeks_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStats {
    pub project_id: i32,
    pub minutes: i32,
    pub hours: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeekData {
    pub year: i32,
    pub week: i32,
    pub total_minutes: i32,
    pub total_hours: f64,
    pub total_amount: f64,
    pub project_stats: Vec<ProjectStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSettings {
    pub enabled: bool,
    pub weekly_goal_hours: Option<f64>,
    pub payment_period_weeks: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLInputData {
    pub timesheets: Vec<TimesheetEntry>,
    pub projects: Vec<Project>,
    pub weeks: Vec<WeekData>,
    pub settings: Settings,
    pub context: Option<Context>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub rate_per_minute: f64,
    #[serde(default)]
    pub project_settings: std::collections::HashMap<i32, ProjectSettings>,
    #[serde(default)]
    pub user_preferences: Option<UserPreferences>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    #[serde(default = "default_sleep_start")]
    pub sleep_start_hour: i32, // 0-23
    #[serde(default = "default_sleep_end")]
    pub sleep_end_hour: i32, // 0-23
    #[serde(default = "default_no_work_hours")]
    pub no_work_before_sleep_hours: i32, // hours before sleep
    #[serde(default = "default_work_on_weekends")]
    pub work_on_weekends: bool,
    #[serde(default)]
    pub project_goals: std::collections::HashMap<i32, f64>, // project_id -> weekly_goal_hours
}

fn default_sleep_start() -> i32 { 0 }
fn default_sleep_end() -> i32 { 8 }
fn default_no_work_hours() -> i32 { 2 }
fn default_work_on_weekends() -> bool { false }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub target_week: Option<i32>,
    pub target_year: Option<i32>,
    pub target_project_id: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastingOutput {
    pub weekly_hours: f64,
    #[serde(default)]
    pub weekly_hours_by_project: std::collections::HashMap<i32, f64>,
    pub monthly_hours: f64,
    pub confidence: f64,
    pub trend: String, // "increasing" | "decreasing" | "stable"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyOutput {
    pub entry_id: i32,
    pub r#type: String, // "duration" | "time" | "pattern" | "project"
    pub severity: String, // "low" | "medium" | "high"
    pub reason: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationOutput {
    pub r#type: String, // "time_allocation" | "project_priority" | "schedule_optimization"
    pub priority: String, // "low" | "medium" | "high"
    pub title: String,
    pub description: String,
    pub action_items: Vec<String>,
    pub expected_impact: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalWorkHours {
    pub start: i32,
    pub end: i32,
    pub days: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakRecommendations {
    pub optimal_break_duration: i32,
    pub break_frequency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductivityOutput {
    pub optimal_work_hours: OptimalWorkHours,
    pub efficiency_by_time: Vec<EfficiencyPoint>,
    pub break_recommendations: BreakRecommendations,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyPoint {
    pub hour: i32,
    pub efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLOutputData {
    pub forecasting: Option<ForecastingOutput>,
    pub anomalies: Option<Vec<AnomalyOutput>>,
    pub recommendations: Option<Vec<RecommendationOutput>>,
    pub productivity: Option<ProductivityOutput>,
}

