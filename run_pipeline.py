# run_pipeline.py
import subprocess
import time
import webbrowser
from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == "__main__":
    tracking_uri = Client().active_stack.experiment_tracker.get_tracking_uri()

    if tracking_uri.startswith("file:"):
        backend_store_path = tracking_uri
    else:
        raise ValueError(f"Неподдерживаемый tracking URI для MLflow UI: {tracking_uri}")
    subprocess.Popen([
        "mlflow", "ui",
        "--backend-store-uri", backend_store_path,
        "--port", "5000"
    ])

    webbrowser.open("http://127.0.0.1:5000")
    time.sleep(3)
    print("MLflow Tracking URI:", tracking_uri)
    training_pipeline(data_path="D:/Credit_Score_MLops/data/data.csv")
