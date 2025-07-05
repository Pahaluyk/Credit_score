# Credit Score Classification (CatBoost ML Pipeline)

📊 Классический ML-проект по предсказанию кредитного рейтинга (Credit Score) клиентов с использованием CatBoostClassifier.

## 📌 Цель проекта

Построить модель, способную классифицировать клиентов по уровню кредитного риска (Poor, Standard, Good) на основе табличных данных с демографией, финансовыми показателями и поведением по займам.

## 🧠 Используемые технологии

- Python (pandas, numpy, scikit-learn)
- CatBoost (GPU)
- Matplotlib / Seaborn
- Jupyter Notebook

## ⚙️ Что сделано

- EDA и анализ пропусков
- Обработка категориальных признаков (One-Hot, Label)
- Feature engineering (в т.ч. Credit_History_Age → в месяцы)
- Заполнение пропусков (в т.ч. восстановление по Annual Income)
- Обучение модели CatBoost с использованием:
  - Мультиклассовой классификации
  - GPU-ускорения
  - EarlyStopping
- Оценка метрик на hold-out test (из train):
  - **Accuracy**: 80.4%
  - **Macro F1-score**: 79%
  - **ROC-AUC (OvR)**: 92.2%
- Вывод confusion matrix и classification report
- Feature Importance



