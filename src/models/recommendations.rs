/// Генератор рекомендаций по оптимизации

use std::collections::HashMap;

use crate::types::{MLInputData, RecommendationOutput, Project};

pub struct RecommendationEngine {
    // KMeans не используется, используем простую эвристику
}

impl RecommendationEngine {
    pub fn new() -> Self {
        Self {}
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

        // Подготовка признаков для кластеризации
        let mut project_ids = Vec::new();
        let mut total_hours_vec = Vec::new();
        let mut weekly_hours_vec = Vec::new();

        for project in projects {
            project_ids.push(project.id);
            total_hours_vec.push(project.total_hours);
            weekly_hours_vec.push(project.avg_hours_per_week);
        }

        // Упрощенная кластеризация на основе средних значений
        // Разделяем проекты на группы по размеру (малые/средние/большие)
        let mut clusters = HashMap::new();
        
        if !total_hours_vec.is_empty() {
            // Вычисляем средние значения признаков
            let avg_total_hours: f64 = total_hours_vec.iter().sum::<f64>() / total_hours_vec.len() as f64;
            
            for (idx, project_id) in project_ids.iter().enumerate() {
                let total_hours = total_hours_vec[idx];
                
                // Простая кластеризация: 0 = малые, 1 = средние, 2 = большие
                let cluster = if total_hours < avg_total_hours * 0.5 {
                    0 // Малые проекты
                } else if total_hours > avg_total_hours * 1.5 {
                    2 // Большие проекты
                } else {
                    1 // Средние проекты
                };
                
                clusters.insert(*project_id, cluster);
            }
        } else {
            // Fallback: все в один кластер
            for project_id in project_ids {
                clusters.insert(project_id, 0);
            }
        }
        
        clusters
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
        
        // Учитываем цели по проектам из предпочтений пользователя
        let project_goals: HashMap<i32, f64> = data.settings
            .user_preferences
            .as_ref()
            .map(|prefs| prefs.project_goals.clone())
            .unwrap_or_default();

        if efficiency.len() < 2 {
            return recommendations;
        }
        
        // Если есть цели по проектам, рекомендуем равномерное распределение
        if !project_goals.is_empty() {
            for (project_id, goal_hours) in &project_goals {
                let current_hours = distribution.get(project_id).copied().unwrap_or(0.0);
                let project_name = self.get_project_name(data, *project_id);
                
                if current_hours < *goal_hours * 0.9 {
                    recommendations.push(RecommendationOutput {
                        r#type: "time_allocation".to_string(),
                        priority: "high".to_string(),
                        title: format!("Увеличьте время на проект '{}'", project_name),
                        description: format!(
                            "Текущее время: {:.1} ч/неделю, цель: {:.1} ч/неделю. Рекомендуется равномерное распределение в течение недели.",
                            current_hours, goal_hours
                        ),
                        action_items: vec![
                            format!("Распределите {:.1} часов равномерно по рабочим дням", goal_hours),
                            "Используйте оптимальные часы работы для этого проекта".to_string(),
                        ],
                        expected_impact: format!("Достижение цели по проекту '{}'", project_name),
                        confidence: 0.8,
                    });
                }
            }
            if !recommendations.is_empty() {
                return recommendations;
            }
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

