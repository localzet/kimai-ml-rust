/// Анализ продуктивности

use std::collections::HashMap;
use chrono::DateTime;

use crate::types::{TimesheetEntry, ProductivityOutput, OptimalWorkHours, BreakRecommendations, EfficiencyPoint, UserPreferences};

pub struct ProductivityAnalyzer {
    preferences: Option<UserPreferences>,
}

impl ProductivityAnalyzer {
    pub fn new() -> Self {
        Self {
            preferences: None,
        }
    }

    pub fn with_preferences(preferences: Option<UserPreferences>) -> Self {
        Self { preferences }
    }

    pub fn analyze(&self, entries: &[TimesheetEntry]) -> ProductivityOutput {
        // 1. Анализ по часам дня
        let hourly_efficiency = self.analyze_hourly_efficiency(entries);

        // 2. Анализ по дням недели
        let daily_efficiency = self.analyze_daily_efficiency(entries);

        // 3. Определение оптимальных часов
        let optimal_hours = self.find_optimal_hours(&hourly_efficiency, &daily_efficiency);

        // 4. Рекомендации по перерывам
        let break_recommendations = self.analyze_breaks(entries);

        ProductivityOutput {
            optimal_work_hours: optimal_hours,
            efficiency_by_time: hourly_efficiency,
            break_recommendations,
        }
    }

    fn analyze_hourly_efficiency(&self, entries: &[TimesheetEntry]) -> Vec<EfficiencyPoint> {
        let mut hourly_data: HashMap<i32, (i32, i32)> = HashMap::new(); // (work, total)

        for entry in entries {
            let hour = entry.hour_of_day;
            let duration = entry.duration;

            let (work, total) = hourly_data.entry(hour).or_insert((0, 0));
            *work += duration;
            *total += 60; // час = 60 минут
        }

        let mut efficiency = Vec::new();
        for hour in 0..24 {
            let (work, total) = hourly_data.get(&hour).copied().unwrap_or((0, 0));
            let eff = if total > 0 {
                work as f64 / total as f64
            } else {
                0.0
            };

            efficiency.push(EfficiencyPoint {
                hour,
                efficiency: eff,
            });
        }

        efficiency
    }

    fn analyze_daily_efficiency(&self, entries: &[TimesheetEntry]) -> HashMap<i32, f64> {
        let mut daily_data: HashMap<i32, (i32, std::collections::HashSet<String>)> = HashMap::new();

        for entry in entries {
            let day = entry.day_of_week;
            let duration = entry.duration;

            // Извлекаем дату из begin
            let date_key = entry.begin.split('T').next().unwrap_or("").to_string();

            let (work, days) = daily_data.entry(day).or_insert_with(|| (0, std::collections::HashSet::new()));
            *work += duration;
            days.insert(date_key);
        }

        let mut efficiency = HashMap::new();
        for (day, (work, days)) in daily_data {
            let n_days = days.len().max(1);
            let avg_hours = (work as f64 / 60.0) / n_days as f64;
            efficiency.insert(day, avg_hours);
        }

        efficiency
    }

    fn find_optimal_hours(
        &self,
        hourly_efficiency: &[EfficiencyPoint],
        daily_efficiency: &HashMap<i32, f64>,
    ) -> OptimalWorkHours {
        let prefs = self.preferences.as_ref();
        let sleep_start = prefs.map(|p| p.sleep_start_hour).unwrap_or(0);
        let sleep_end = prefs.map(|p| p.sleep_end_hour).unwrap_or(8);
        let no_work_before_sleep = prefs.map(|p| p.no_work_before_sleep_hours).unwrap_or(2);
        let work_on_weekends = prefs.map(|p| p.work_on_weekends).unwrap_or(false);

        // Фильтруем часы с учетом предпочтений пользователя
        let mut filtered_efficiency: Vec<_> = hourly_efficiency.iter()
            .filter(|e| {
                // Исключаем часы сна
                if e.hour >= sleep_start && e.hour < sleep_end {
                    return false;
                }
                // Исключаем часы перед сном
                if sleep_start > 0 {
                    let no_work_start = (sleep_start - no_work_before_sleep + 24) % 24;
                    if no_work_start <= sleep_start {
                        if e.hour >= no_work_start && e.hour < sleep_start {
                            return false;
                        }
                    } else {
                        if e.hour >= no_work_start || e.hour < sleep_start {
                            return false;
                        }
                    }
                }
                true
            })
            .collect();

        // Сортировка по эффективности
        filtered_efficiency.sort_by(|a, b| b.efficiency.partial_cmp(&a.efficiency).unwrap_or(std::cmp::Ordering::Equal));

        // Топ-8 часов
        let top_hours: Vec<i32> = filtered_efficiency
            .iter()
            .take(8)
            .filter(|e| e.efficiency > 0.0)
            .map(|e| e.hour)
            .collect();

        // Топ дней (исключаем выходные, если не работаем на выходных)
        let mut sorted_days: Vec<_> = daily_efficiency.iter().collect();
        sorted_days.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut top_days: Vec<i32> = sorted_days.iter()
            .filter(|(&day, _)| {
                if !work_on_weekends {
                    // 0 = воскресенье, 6 = суббота
                    day != 0 && day != 6
                } else {
                    true
                }
            })
            .take(5)
            .map(|(&d, _)| d)
            .collect();

        if top_days.is_empty() {
            if work_on_weekends {
                top_days = vec![1, 2, 3, 4, 5, 6, 0];
            } else {
                top_days = vec![1, 2, 3, 4, 5]; // Пн-Пт по умолчанию
            }
        }

        OptimalWorkHours {
            start: top_hours.iter().copied().min().unwrap_or(9),
            end: top_hours.iter().copied().max().unwrap_or(18),
            days: top_days,
        }
    }

    fn analyze_breaks(&self, entries: &[TimesheetEntry]) -> BreakRecommendations {
        let sessions = self.extract_sessions(entries);

        if sessions.is_empty() {
            return BreakRecommendations {
                optimal_break_duration: 15,
                break_frequency: 2.0,
            };
        }

        let avg_session_duration = sessions.iter().map(|s| s.duration).sum::<i32>() as f64 / sessions.len() as f64;

        // Рекомендации на основе средней длительности сессии
        let (break_duration, break_frequency) = if avg_session_duration > 120.0 {
            (15, 2.0) // каждые 2 часа
        } else if avg_session_duration > 60.0 {
            (10, 1.5)
        } else {
            (5, 1.0)
        };

        BreakRecommendations {
            optimal_break_duration: break_duration,
            break_frequency,
        }
    }

    fn extract_sessions(&self, entries: &[TimesheetEntry]) -> Vec<Session> {
        // Группировка по дням
        let mut daily_entries: HashMap<String, Vec<&TimesheetEntry>> = HashMap::new();
        for entry in entries {
            if let Some(date_key) = entry.begin.split('T').next() {
                daily_entries.entry(date_key.to_string()).or_insert_with(Vec::new).push(entry);
            }
        }

        let mut sessions = Vec::new();

        for (_, day_entries) in daily_entries {
            // Сортировка по времени начала
            let mut sorted: Vec<_> = day_entries.iter().collect();
            sorted.sort_by_key(|e| &e.begin);

            // Объединение близких записей в сессии
            let mut current_session = Session {
                start: sorted[0].begin.clone(),
                end: sorted[0].end.clone().unwrap_or_else(|| sorted[0].begin.clone()),
                duration: sorted[0].duration,
            };

            for entry in sorted.iter().skip(1) {
                // Если перерыв < 30 минут, считаем продолжением сессии
                if let (Ok(current_end), Ok(next_start)) = (
                    DateTime::parse_from_rfc3339(&current_session.end),
                    DateTime::parse_from_rfc3339(&entry.begin),
                ) {
                    let gap = (next_start - current_end).num_minutes();

                    if gap < 30 {
                        current_session.end = entry.end.clone().unwrap_or_else(|| entry.begin.clone());
                        current_session.duration += entry.duration;
                    } else {
                        sessions.push(current_session);
                        current_session = Session {
                            start: entry.begin.clone(),
                            end: entry.end.clone().unwrap_or_else(|| entry.begin.clone()),
                            duration: entry.duration,
                        };
                    }
                }
            }

            sessions.push(current_session);
        }

        sessions
    }
}

struct Session {
    start: String,
    end: String,
    duration: i32,
}

