import sys
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logger.info("=" * 50)
            logger.info(">>>> Starting Training Pipeline Execution <<<<")
            logger.info("=" * 50)

            # Step 1: Data Ingestion
            logger.info(">>> Stage 1: Data Ingestion Started <<<")
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logger.info(f"Ingestion Finished. Train Path: {train_path}, Test Path: {test_path}")

            # Step 2: Data Validation
            logger.info(">>> Stage 2: Data Validation Started <<<")
            validation = DataValidation()
            validation_status = validation.validate_all_columns(train_path)
            
            if not validation_status:
                raise Exception("Data Validation failed! Aborting pipeline.")
            logger.info(f"Data Validation Passed: {validation_status}")

            # Step 3: Data Transformation
            logger.info(">>> Stage 3: Data Transformation Started <<<")
            transformation = DataTransformation()
            transformed_train_path, transformed_test_path, scaler_path = transformation.initiate_data_transformation(train_path, test_path)
            logger.info(f"Transformation Finished. Transformed Train Path: {transformed_train_path}, Transformed Test Path: {transformed_test_path}")

            # Step 4: Model Training
            logger.info(">>> Stage 4: Model Training Started <<<")
            trainer = ModelTrainer()
            model_path = trainer.initiate_model_trainer(transformed_train_path)
            logger.info(f"Model Training Finished. Saved Model Path: {model_path}")

            # Step 5: Model Evaluation (تقييم الموديل على بيانات الـ Test المعالجة)
            logger.info(">>> Stage 5: Model Evaluation Started <<<")
            evaluation = ModelEvaluation()
            metrics = evaluation.initiate_model_evaluation(model_path, transformed_test_path)
            logger.info(f"Model Evaluation Finished. Final Metrics: {metrics}")

            logger.info("=" * 50)
            logger.info(">>>> Training Pipeline Completed Successfully! <<<<")
            logger.info("=" * 50)

        except Exception as e:
            logger.error("Training Pipeline failed at one of the stages.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()