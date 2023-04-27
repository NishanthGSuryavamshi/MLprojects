from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
import sys
from src.utils2 import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        'artifacts_2', 'preprocessor_obj.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transforer_object(self):
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ['gender', 'race_ethnicity',
                                    'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='median')),
                       ('scaler', StandardScaler())

                       ]
            )
            cat_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                       ('oh encoder', OneHotEncoder()),
                       ('scaler', StandardScaler(with_mean=False))

                       ]
            )
            logging.info(
                'num pipeline and categorical pipeline created and transformation pipeline starting')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('categorical_pipeline', cat_pipeline, categorical_features)

                ]
            )
            logging.info('transformation done')
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('reading train & test data completed')
            logging.info('obtaining processor obj')

            preprocessor_obj = self.get_data_transforer_object()
            target_column = 'math_score'
            numerical_feature = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(
                columns=target_column, axis=1)
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=target_column, axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(
                'train input and target, test input and target seperated')
            logging.info(
                'applying preprocessor initiate obj on train and test input features')

            input_train_arr = preprocessor_obj.fit_transform(
                input_feature_train_df)
            input_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info(
                'transformation applied on train and test input features')
            logging.info(
                'concatinating train dep and indep features,concatinating test dep and indep features,')

            train_arr = np.c_[
                (input_feature_train_df, np.array(target_feature_train_df))]
            test_arr = np.c_[
                (input_feature_test_df, np.array(target_feature_test_df))]

            logging.info('concatination done')

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info('obj saved ')
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
