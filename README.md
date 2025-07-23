Credit_Score_MLOps/
├── .venv/ # Виртуальное окружение
├── .zen/ # Конфигурации ZenML
├── catboost_logs/ # Логи CatBoost
├── data/ # Данные для обучения
│ └── data.csv
├── ipynb_checkpoints/ # Автоматические чекпойнты Jupyter
│ └── Credit_score.ipynb
├── model/ # Модули модели и метрики
│ ├── data_cleaning.py
│ ├── evaluation.py
│ └── model_dev.py
├── pipelines/ # Конфигурация ZenML пайплайна
│ └── training_pipeline.py
├── saved_model/ # Финальная обученная модель
│ └── credit_score_model.cbm
├── steps/ # Отдельные шаги пайплайна
│ ├── ingest_data.py
│ ├── clean_data.py
│ ├── model_train.py
│ └── evaluation.py
├── zenml_artifact_store/ # Локальное хранилище артефактов ZenML
├── .gitignore # Файл исключений Git
├── README.md # Документация проекта
├── requirements.txt # Зависимости Python
├── run_pipeline.py # Скрипт запуска пайплайна
└── setup_zenml.py # Скрипт настройки ZenML стека
