import sys
from leukemia.exception.exception import CustomException
from leukemia.logging.logger import logging 
from leukemia.data.ingestion import DataIngestion
from leukemia.data.transformation import DataTransformation
from leukemia.data.validation import DataValidation
from leukemia.entity.model_architecture import LeukemiaCNN, save_model, load_model
from leukemia.train.trainer import Train
from leukemia.entity.artifact_entity import DataTransformationArtifact, Predict_image
from leukemia.utils.visualization import plot_graph
from leukemia.nlp.generator.report_generator import LeukemiaReportGenerator
from leukemia.inference.predict import Predictor
from leukemia.constant.training_pipeline import IMAGE_PATH

if __name__ == "__main__":
    try:
        dataingestion = DataIngestion()
        data_ingestion_artifact = dataingestion.initiate_data_ingestion()
        logging.info("Data Ingestion complete")

        datatransformation = DataTransformation(data_ingestion_artifact)
        data_transformation_artifact = datatransformation.initiate_data_transformation()
        logging.info("Data Transformation Complete")

        model = LeukemiaCNN()
        logging.info(f"Model initialised {model}")

        datavalidation = DataValidation()

        trainer = Train(data_validation=datavalidation, model=model)

        model_results = trainer.train_model(model=model,
                                            train_dataloader=data_transformation_artifact.train_dataloader,
                                            val_dataloader=data_transformation_artifact.val_dataloader)

        logging.info("Saving the model")
        save_model(model=model_results)

        logging.info("Training complete")
        print(f"\nTraining Summary:")
        print(f"Best Val Accuracy: {max(model_results['val_acc']):.2f}%")
        print(f"Best Val ROC-AUC: {max(model_results['val_roc_auc']):.4f}")
        logging.info("Displaying model results")
        plot_graph(model_results)

        logging.info("loading the model")
        load_model()

        predictor = Predictor(model=model)

        test_image_path = IMAGE_PATH
        predict_result = predictor.predict_image(
            model=model,
            image_path=str(test_image_path),
            class_name=['hem', 'all']
        )

        logging.info("Generating Medical Report")
        generator = LeukemiaReportGenerator()
        medical_report = generator.generate_report(
            prediction=predict_result.prediction,
            confidence=predict_result.confidence
        )
        print(medical_report)
    except Exception as e:
        raise CustomException(str(e), e)