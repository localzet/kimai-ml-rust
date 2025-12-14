/// Генератор рекомендаций по оптимизации

use ndarray::{Array1, Array2};
use linfa::prelude::*;
use linfa_clustering::KMeans;
use std::collections::HashMap;

use crate::types::{MLInputData, RecommendationOutput, Project};

pub struct RecommendationEngine {
    kmeans: Option<KMeans<f64, bool>>,
}

impl RecommendationEngine {
    pub fn new() -> Self {
        Self { kmeans: None }
    }

    pub fn generate_recommendations(&mut self, data: &MLInputData) -> Vec<RecommendationOutput> {
        let mut recommendations = Vec::new();

        // 1. Анализ эффективности проектов
        let project_efficiency = self.calculate_project_efficiency(data);

        // 2. Кластеризация проектов
        let _project_clusters = self.cluster_projects(&data.projects);

        // 3. Анализ распределения времени
        let time_distribution = self.analyze_time_distribution(&data.weeks);

        // 4. Генерация рекомендаций
        recommendations.extend(self.recommend_time_allocation(&project_efficiency, &time_distribution, data));
        recommendations.extend(self.recommend_project_priority(&project_efficiency, data));
        recommendations.extend(self.recommend_schedule_optimization(data));

        recommendations
    }

    fn calculate_project_efficiency(&self, data: &MLInputData) -> HashMap<i32, f64> {
        let mut efficiency = HashMap::new();
        let rate_per_hour = data.settings.rate_per_minute * 60.0;

        for project in &data.projects {
            if project.total_hours > 0.0 {
                let total_amount = project.total_hours * rate_per_hour;
                efficiency.insert(project.id, total_amount / project.total_hours);
            } else {
                efficiency.insert(project.id, 0.0);
            }
        }

        efficiency
    }

    fn cluster_projects(&mut self, projects: &[Project]) -> HashMap<i32, usize> {
        if projects.len() < 3 {
            return projects.iter().map(|p| (p.id, 0)).collect();
        }

        // Подготовка признаков
        let mut features = Vec::new();
        let mut project_ids = Vec::new();

        for project in projects {
            features.push(vec![
                project.total_hours,
                project.avg_hours_per_week,
                project.weeks_count as f64,
            ]);
            project_ids.push(project.id);
        }

        // Нормализация
        let n_features = 3;
        let n_samples = features.len();
        let mut features_array = Array2::zeros((n_samples, n_features));

        for (i, feat) in features.iter().enumerate() {
            for (j, val) in feat.iter().enumerate() {
                features_array[[i, j]] = *val;
            }
        }

        // Среднее и стандартное отклонение
        let mean: Vec<f64> = (0..n_features)
            .map(|j| {
                (0..n_samples).map(|i| features_array[[i, j]]).sum::<f64>() / n_samples as f64
            })
            .collect();

        let std: Vec<f64> = (0..n_features)
            .map(|j| {
                let variance = (0..n_samples)
                    .map(|i| (features_array[[i, j]] - mean[j]).powi(2))
                    .sum::<f64>()
                    / n_samples as f64;
                variance.sqrt().max(1e-10)
            })
            .collect();

        // Нормализация
        for i in 0..n_samples {
            for j in 0..n_features {
                features_array[[i, j]] = (features_array[[i, j]] - mean[j]) / std[j];
            }
        }

        // Кластеризация через KMeans
        let n_clusters = projects.len().min(3);
        if n_clusters > 0 && n_samples > 0 {
            // Создаем dataset для кластеризации
            // KMeans требует правильный формат данных
            let dataset = Dataset::new(features_array.clone(), Array1::zeros(n_samples));
            
            match KMeans::params(n_clusters).fit(&dataset) {
                Ok(model) => {
                    let clusters = model.predict(&dataset);
                    project_ids
                        .iter()
                        .zip(clusters.targets().iter())
                        .map(|(&id, cluster)| (id, *cluster as usize))
                        .collect()
                }
                Err(_) => {
                    // Fallback: все в один кластер
                    project_ids.iter().map(|&id| (id, 0)).collect()
                }
            }
        } else {
            project_ids.iter().map(|&id| (id, 0)).collect()
        }
    }

    fn analyze_time_distribution(&self, weeks: &[crate::types::WeekData]) -> HashMap<i32, f64> {
        let mut distribution = HashMap::new();

        for week in weeks {
            for stat in &week.project_stats {
                *distribution.entry(stat.project_id).or_insert(0.0) += stat.hours;
            }
        }

        // Нормализация к среднему за неделю
        if !weeks.is_empty() {
            let n_weeks = weeks.len() as f64;
            for hours in distribution.values_mut() {
                *hours /= n_weeks;
            }
        }

        distribution
    }

    fn recommend_time_allocation(
        &self,
        efficiency: &HashMap<i32, f64>,
        distribution: &HashMap<i32, f64>,
        data: &MLInputData,
    ) -> Vec<RecommendationOutput> {
        let mut recommendations = Vec::new();

        if efficiency.len() < 2 {
            return recommendations;
        }

        // Сортировка по эффективности
        let mut sorted: Vec<_> = efficiency.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((&top_project_id, &efficiency_val)) = sorted.first() {
            if efficiency_val > 0.0 {
                let current_hours = distribution.get(&top_project_id).copied().unwrap_or(0.0);

                if current_hours > 0.0 {
                    let recommended_hours = current_hours * 1.2;
                    let project_name = self.get_project_name(data, top_project_id);

                    recommendations.push(RecommendationOutput {
                        r#type: "time_allocation".to_string(),
                        priority: "high".to_string(),
                        title: "Увеличьте время на высокоэффективные проекты".to_string(),
                        description: format!("Проект '{}' показывает высокую эффективность", project_name),
                        action_items: vec![
                            format!("Увеличьте время на проект до {:.1} часов/неделю", recommended_hours),
                            "Перераспределите 15-20% времени с менее эффективных проектов".to_string(),
                        ],
                        expected_impact: "Потенциальное увеличение дохода на 10-15%".to_string(),
                        confidence: 0.75,
                    });
                }
            }
        }

        recommendations
    }

    fn recommend_project_priority(
        &self,
        efficiency: &HashMap<i32, f64>,
        data: &MLInputData,
    ) -> Vec<RecommendationOutput> {
        let mut recommendations = Vec::new();

        if efficiency.len() < 2 {
            return recommendations;
        }

        let mut sorted: Vec<_> = efficiency.iter().collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

        let low_efficiency: Vec<_> = sorted.iter().take(3).filter(|(_, &eff)| eff > 0.0).collect();

        if let Some((&project_id, _)) = low_efficiency.first() {
            let project_name = self.get_project_name(data, project_id);

            recommendations.push(RecommendationOutput {
                r#type: "project_priority".to_string(),
                priority: "medium".to_string(),
                title: "Пересмотрите приоритеты проектов".to_string(),
                description: "Некоторые проекты показывают низкую эффективность".to_string(),
                action_items: vec![
                    format!("Проанализируйте проект '{}'", project_name),
                    "Рассмотрите возможность перераспределения времени".to_string(),
                ],
                expected_impact: "Оптимизация использования времени".to_string(),
                confidence: 0.6,
            });
        }

        recommendations
    }

    fn recommend_schedule_optimization(&self, data: &MLInputData) -> Vec<RecommendationOutput> {
        let mut recommendations = Vec::new();

        if data.timesheets.is_empty() {
            return recommendations;
        }

        // Анализ распределения по часам
        let mut hourly_distribution: HashMap<i32, i32> = HashMap::new();
        for entry in &data.timesheets {
            *hourly_distribution.entry(entry.hour_of_day).or_insert(0) += entry.duration;
        }

        if !hourly_distribution.is_empty() {
            let mut sorted: Vec<_> = hourly_distribution.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            let top_hours: Vec<String> = sorted.iter().take(3).map(|(&h, _)| h.to_string()).collect();

            recommendations.push(RecommendationOutput {
                r#type: "schedule_optimization".to_string(),
                priority: "medium".to_string(),
                title: "Оптимизируйте расписание работы".to_string(),
                description: format!("Наиболее продуктивные часы: {}:00", top_hours.join(", ")),
                action_items: vec![
                    format!("Планируйте важные задачи на {}:00", top_hours[0]),
                    "Используйте менее продуктивные часы для рутинных задач".to_string(),
                ],
                expected_impact: "Улучшение продуктивности на 10-15%".to_string(),
                confidence: 0.7,
            });
        }

        recommendations
    }

    fn get_project_name(&self, data: &MLInputData, project_id: i32) -> String {
        data.projects
            .iter()
            .find(|p| p.id == project_id)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| format!("Проект {}", project_id))
    }
}

impl Default for RecommendationEngine {
    fn default() -> Self {
        Self::new()
    }
}

