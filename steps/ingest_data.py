import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        df = pd.read_csv(self.data_path, dtype={'Monthly_Balance': str}, low_memory=False)
        df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'], errors='coerce')
        df['Monthly_Balance'] = df['Monthly_Balance'].fillna(df['Monthly_Balance'].mean())
        
        return df

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    try:
        ingest_data_obj = IngestData(data_path)
        df = ingest_data_obj.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        return e
