# 🏦 Credit Score Prediction — Проект с использованием ZenML и MLflow

Этот проект представляет собой **end-to-end MLOps пайплайн** для задачи **предсказания кредитного рейтинга** клиентов.  
Он охватывает все ключевые этапы жизненного цикла ML-проекта:

- 📥 Загрузка и очистка данных
- 🧠 Обучение модели CatBoost
- 📊 Логгирование метрик и параметров с помощью MLflow
- 🧪 Оценка производительности модели
- 💾 Сохранение модели и генерация feature importance
- 🧱 Интеграция в ZenML пайплайн
- 🔬 Повторяемость и масштабируемость экспериментов

Цель — построить модульную, расширяемую и отслеживаемую ML-систему, которую легко поддерживать и развивать.

---

## 📊 Используемые метрики

Ниже приведены метрики, которые автоматически логгируются в MLflow при каждой итерации обучения модели:

<img width="1899" height="1034" alt="image" src="https://github.com/user-attachments/assets/7bf0b4b7-808c-43c6-ba89-4e374373d6d2" />

Также в MLflow выводится Feature Importance

<img width="1561" height="707" alt="image" src="https://github.com/user-attachments/assets/9e64aa8d-e34e-406c-8608-cfb7711b9966" />
> Все метрики автоматически логгируются в MLflow и могут быть отслежены в интерфейсе [`http://127.0.0.1:5000`](http://127.0.0.1:5000) при запуске `mlflow ui`.

---
## Пайплайн ZenML

<img width="848" height="586" alt="image" src="https://github.com/user-attachments/assets/87035329-d023-4e91-b1fd-4ee4b91c94fc" />


# Запуск проекта с ZenML и MLflow

## 1. Создать и активировать виртуальное окружение

# 🔧 Установка и запуск проекта

# 1. Создание и активация виртуального окружения
python -m venv .venv
.venv\Scripts\activate  # для Windows
# source .venv/bin/activate  # для Linux/MacOS

# 2. Установка зависимостей проекта
pip install -r requirements.txt

# 3. Установка ZenML с серверной поддержкой
pip install "zenml[server]"

# 4. Установка интеграции MLflow
zenml integration install mlflow -y

# 5. Настройка ZenML (создание артефакт-хранилища и стека)
python setup_zenml.py

# 6. Запуск ML-пайплайна
python run_pipeline.py

# 7. Вход в локальный ZenML UI (откроется автоматически)
zenml login --local --blocking

## 📁 Структура проекта

Credit_Score_MLOps/  
├── .venv/                      - Виртуальное окружение  
├── .zen/                       - Конфигурации ZenML  
├── catboost_logs/             - Логи CatBoost  
├── data/                      - Данные для обучения  
│   └── data.csv  
├── ipynb_checkpoints/         - Автоматические чекпойнты Jupyter  
│   └── Credit_score.ipynb  
├── model/                     - Модули модели и метрики  
│   ├── data_cleaning.py  
│   ├── evaluation.py  
│   └── model_dev.py  
├── pipelines/                 - Конфигурация ZenML пайплайна  
│   └── training_pipeline.py  
├── saved_model/               - Финальная обученная модель  
│   └── credit_score_model.cbm  
├── steps/                     - Отдельные шаги пайплайна  
│   ├── ingest_data.py  
│   ├── clean_data.py  
│   ├── model_train.py  
│   └── evaluation.py  
├── zenml_artifact_store/      - Локальное хранилище артефактов ZenML  
├── .gitignore                 - Файл исключений Git  
├── README.md                  - Документация проекта  
├── requirements.txt           - Зависимости Python  
├── run_pipeline.py            - Скрипт запуска пайплайна  
└── setup_zenml.py             - Скрипт настройки ZenML стека  
