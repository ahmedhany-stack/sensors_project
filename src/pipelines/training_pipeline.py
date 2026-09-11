import os
import sys
import yaml
import onnx
from onnx import compose

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class TrainingPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)

    def export_combined_onnx_pipeline(
        self, 
        scaler_onnx_path: str = "models/scaler.onnx", 
        model_onnx_path: str = "models/xgb_rul_model.onnx", 
        output_onnx_path: str = "models/full_rul_pipeline.onnx"
    ):
        """
        دمج ملف الـ Scaler وملف الـ XGBoost باستخدام compose.add_prefix
        لتفادي تعارض الأسماء و اختلاف الـ IR Versions.
        """
        try:
            logger.info("Starting ONNX Graph merging directly from .onnx files...")

            # 1. تحميل الملفين
            scaler_model = onnx.load(scaler_onnx_path)
            regressor_model = onnx.load(model_onnx_path)

            # 2. توحيد الـ IR Version
            target_ir_version = max(scaler_model.ir_version, regressor_model.ir_version)
            scaler_model.ir_version = target_ir_version
            regressor_model.ir_version = target_ir_version

            # 3. إضافة Prefix للنموذج الثاني لتجنب تكرار أسماء النودز
            regressor_model = compose.add_prefix(regressor_model, prefix="xgb_")

            # 4. استخراج اسم مخرج الـ Scaler واسم مدخل الـ XGBoost بعد إضافة الـ Prefix
            scaler_output_name = scaler_model.graph.output[0].name
            model_input_name = regressor_model.graph.input[0].name

            logger.info(f"Connecting Scaler Output [{scaler_output_name}] -> Model Input [{model_input_name}]")

            # 5. دمج الـ Models
            combined_onnx = compose.merge_models(
                scaler_model,
                regressor_model,
                io_map=[(scaler_output_name, model_input_name)]
            )

            # 6. حفظ ملف الـ ONNX المدمج النهائي
            os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
            onnx.save(combined_onnx, output_onnx_path)

            logger.info(f"Combined Scaler + Model ONNX saved successfully to: {output_onnx_path}")
            return output_onnx_path

        except Exception as e:
            logger.error("Failed to export combined ONNX pipeline.")
            raise CustomException(e, sys)

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
                raise ValueError("Data Validation failed! Aborting pipeline.")
            logger.info(f"Data Validation Passed: {validation_status}")

            # Step 3: Data Transformation
            logger.info(">>> Stage 3: Data Transformation Started <<<")
            transformation = DataTransformation()
            transformed_train_path, transformed_test_path, scaler_onnx_path = transformation.initiate_data_transformation(train_path, test_path)
            logger.info(f"Transformation Finished. Transformed Train Path: {transformed_train_path}, Transformed Test Path: {transformed_test_path}")

            # Step 4: Model Training
            logger.info(">>> Stage 4: Model Training Started <<<")
            trainer = ModelTrainer()
            model_onnx_path = trainer.initiate_model_trainer(transformed_train_path)
            logger.info(f"Model Training Finished. Saved Model Path: {model_onnx_path}")

            # Step 4.5: Direct ONNX Graph Merge (مع إضافة Prefix)
            combined_onnx_path = self.export_combined_onnx_pipeline(
                scaler_onnx_path="models/scaler.onnx",
                model_onnx_path="models/xgb_rul_model.onnx",
                output_onnx_path="models/full_rul_pipeline.onnx"
            )

            # Step 5: Model Evaluation
            logger.info(">>> Stage 5: Model Evaluation Started <<<")
            evaluation = ModelEvaluation()
            metrics = evaluation.initiate_model_evaluation(combined_onnx_path, transformed_test_path)
            logger.info(f"Model Evaluation Finished. Final Metrics: {metrics}")

            logger.info("=" * 50)
            logger.info(">>>> Training Pipeline Completed Successfully! <<<<")
            logger.info("=" * 50)

            return metrics

        except Exception as e:
            logger.error("Training Pipeline failed at one of the stages.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()