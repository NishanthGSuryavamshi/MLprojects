import sys
import os
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts_2', 'train_1.csv')
    test_data_path: str = os.path.join('artifacts_2', 'test_.csv')
    raw_data_path: str = os.path.join('artifacts_2', 'raw_.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info('data ingestion initiated')
        try:
            df = pd.read_csv('Notebook\Data\StudentsPerformance.csv')
            logging.info('read the data as dataframe')

            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.train_data_path)
            logging.info('train_test_split initiated')

            train_set, test_set = train_test_split(
                df, test_size=.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path)
            test_set.to_csv(self.ingestion_config.test_data_path)
            logging.info('train and test dataset saved in artifacts folder')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
