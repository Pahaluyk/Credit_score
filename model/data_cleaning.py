import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:

        try:
            data.replace('_______' , np.nan, inplace=True)
            data.replace('_', np.nan, inplace=True)
            data.replace('!@9#%8', np.nan, inplace=True)
            data.replace('__10000__', np.nan, inplace=True)
                
            def safe_float_cast(val):
                val_str = str(val).replace("_", "").strip()
                if val_str == "":
                    return np.nan
                try:
                    return float(val_str)
                except (ValueError, TypeError):
                    return np.nan

            numeric_cols = [
                'Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
                'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance'
            ]

            for col in numeric_cols:
                data[col] = data[col].map(safe_float_cast)

            data['Payment_of_Min_Amount'].replace("NM", "No", inplace=True)
            data['Num_Bank_Accounts'] = data['Num_Bank_Accounts'].replace(-1, 1)
            data = data[data['Num_Credit_Card'] <= 15]
            for df in [data]:
                df['Occupation'] = df['Occupation'].fillna('Other')
                df.drop(df[df['Age'] > 100].index, inplace=True)
                df.drop(df[df['Age'] < 1].index, inplace=True)

                df.loc[df['Num_of_Loan'] < 0, 'Num_of_Loan'] = np.nan
                df.loc[df['Num_of_Loan'] > 9, 'Num_of_Loan'] = np.nan

                df.loc[df['Num_Credit_Card'] > 11, 'Num_Credit_Card'] = np.nan

                df.loc[df['Interest_Rate'] > 34, 'Interest_Rate'] = np.nan

                df.loc[df['Annual_Income'] > 300000, 'Annual_Income'] = np.nan

                df.loc[df['Total_EMI_per_month'] > 5000, 'Total_EMI_per_month'] = np.nan

                df.loc[df['Num_Bank_Accounts'] > 100, 'Num_Bank_Accounts'] = np.nan
                df.loc[df['Num_Bank_Accounts'] < 0, 'Num_Bank_Accounts'] = np.nan

                df.loc[df['Num_of_Delayed_Payment'] > 100, 'Num_of_Delayed_Payment'] = np.nan

                df.loc[0, 'Occupation'] = 'Scientist'
            
            data["Interest_Rate"].fillna(data["Interest_Rate"].median(), inplace=True)
            data["Num_of_Loan"].fillna(data["Num_of_Loan"].median(), inplace=True)
            data["Total_EMI_per_month"].fillna(data["Total_EMI_per_month"].median(), inplace=True)
            data['Annual_Income'].fillna(data['Annual_Income'].median(), inplace=True)
            columns_to_fill_with_mean = [
                "Changed_Credit_Limit",
                "Monthly_Inhand_Salary",
                "Num_of_Delayed_Payment",
                "Amount_invested_monthly",
                "Monthly_Balance",
                "Num_Credit_Inquiries"
            ]
            mapping_mix = {'Bad': 0, 'Standard': 1, 'Good': 2}
            mapping_score = {'Poor': 0, 'Standard': 1, 'Good': 2}
            data['Credit_Mix'] = data['Credit_Mix'].map(mapping_mix)
            data['Credit_Score'] = data['Credit_Score'].map(mapping_score)
            for col in columns_to_fill_with_mean:
                data[col].fillna(data[col].mean(), inplace=True)
            mode_creditmix = data.groupby('Customer_ID')["Credit_Mix"].transform(
                lambda x: x.mode()[0] if not x.mode().empty else np.nan
            )
            mode_global = data["Credit_Mix"].mode()[0]

            data["Credit_Mix"] = data["Credit_Mix"].fillna(mode_creditmix.fillna(mode_global))
            mode_age = data.groupby('Customer_ID')["Age"].transform(
                lambda x: x.mode()[0] if not x.mode().empty else np.nan
            )
            mode_global = data["Age"].mode()[0]

            data["Age"] = data["Age"].fillna(mode_age.fillna(mode_global))
            data.drop(columns=[
            "ID", "Name", "SSN", "Customer_ID"
            ],
            inplace=True)
            
            
            
            payment_behaviour_mapping = {
                'High_spent_Small_value_payments': 0,
                'Low_spent_Large_value_payments': 1,
                'Low_spent_Medium_value_payments': 2,
                'Low_spent_Small_value_payments': 3,
                'High_spent_Medium_value_payments': 4,
                'High_spent_Large_value_payments': 5,
            }

            data['Payment_Behaviour'] = data['Payment_Behaviour'].map(payment_behaviour_mapping)
            
            
            
            mapping = {'No': 0, 'Yes': 1}

            data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].map(mapping)
            
            
            
            def encode_column(data, column_name):
                data = pd.get_dummies(data, columns=[column_name], drop_first=True)
                return data

            data = encode_column(data, 'Occupation')
                
            logging.info(f"Всего строк: {df.shape[0]}")
            for col in ["Type_of_Loan", "Month", "Credit_History_Age"]:
                data[col] = data[col].fillna("Unknown")
            return data
        except Exception as e:
            logging.error(e)
            raise e


class DataDivideStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:

        try:
            X = data.drop("Credit_Score", axis=1)
            y = data["Credit_Score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        return self.strategy.handle_data(self.data)

