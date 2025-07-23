import os
from zenml.client import Client

root_path = os.path.abspath("./artifact_store")

# Регистрируем хранилище с абсолютным путем
os.system(f'zenml artifact-store register artifact_store --path="{root_path}" --flavor=local')

# Регистрируем трекер
os.system('zenml experiment-tracker register mlflow_tracker --flavor=mlflow')

# Регистрируем стек
os.system('zenml stack register mlflow_stack -a artifact_store -o default -d mlflow -e mlflow_tracker --set')
