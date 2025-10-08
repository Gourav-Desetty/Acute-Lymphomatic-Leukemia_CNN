import sys
from leukemia.exception.exception import CustomException
from leukemia.logging.logger import logging 
from leukemia.data.ingestion import DataIngestion
from leukemia.data.transformation import DataTransformation

if __name__ == "__main__":
    try:
        dataingestion = DataIngestion()
        data_ingestion_artifact = dataingestion.initiate_data_ingestion()
        logging.info("Data Ingestion complete")

        datatransformation = DataTransformation(data_ingestion_artifact)
        data_transformation_artifact = datatransformation.initiate_data_transformation()
        logging.info("Data Transformation Complete")

    except Exception as e:
        raise CustomException(e, sys)