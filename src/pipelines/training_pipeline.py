import os
import sys
import yaml
import onnx
from onnx import compose
import mlflow
import mlflow.onnx
from mlflow.tracking import MlflowClient

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
        
        # استخراج إعدادات MLflow
        trainer_cfg = self.config.get("model_trainer", {})
        mlflow_cfg = trainer_cfg.get("mlflow", {})
        self.tracking_uri = mlflow_cfg.get("tracking_uri", "http://127.0.0.1:5000")
        self.experiment_name = mlflow_cfg.get("experiment_name", "RUL_Prediction")

    def export_combined_onnx_pipeline(
        self, 
        scaler_onnx_path: str = "models/scaler.onnx", 
        model_onnx_path: str = "models/xgb_rul_model.onnx", 
        output_onnx_path: str = "models/full_rul_pipeline.onnx",
        registered_model_name: str = "RUL_XGBoost_Model",
        alias: str = "Production"
    ):
        """
        دمج ملف الـ Scaler وملف الـ XGBoost باستخدام compose.add_prefix
        لتفادي تعارض الأسماء، ضبط الـ IR Version بما يتوافق مع Triton Server، وتسجيل الموديل المدمج في MLflow Model Registry.
        """
        try:
            logger.info("Starting ONNX Graph merging directly from .onnx files...")

            # 1. تحميل الملفين
            scaler_model = onnx.load(scaler_onnx_path)
            regressor_model = onnx.load(model_onnx_path)

            # 2. توحيد وضبط الـ IR Version بحيث لا يتجاوز 9 ليكون متوافقاً مع Triton Server
            target_ir_version = min(max(scaler_model.ir_version, regressor_model.ir_version), 9)
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

            # 6. حفظ ملف الـ ONNX المدمج النهائي محلياً
            os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
            onnx.save(combined_onnx, output_onnx_path)
            logger.info(f"Combined Scaler + Model ONNX saved successfully to: {output_onnx_path}")

            # -------------------------------------------------------------
            # 7. MLOps Best Practice: تسجيل الموديل المدمج في MLflow Model Registry
            # -------------------------------------------------------------
            logger.info("Registering Combined ONNX Pipeline into MLflow Model Registry...")
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run(run_name="Full_Pipeline_ONNX_Registration"):
                # تسجيل الموديل في الـ Artifacts وإضافته للـ Model Registry
                model_info = mlflow.onnx.log_model(
                    onnx_model=combined_onnx,
                    artifact_path="full_onnx_pipeline",
                    registered_model_name=registered_model_name
                )

            # تعيين الـ Alias المطلوبة (Production) على أحدث إصدار مسجل
            client = MlflowClient(tracking_uri=self.tracking_uri)
            latest_version = model_info.registered_model_version

            client.set_registered_model_alias(
                name=registered_model_name,
                alias=alias,
                version=latest_version
            )

            logger.info(
                f"Successfully registered model '{registered_model_name}' "
                f"(Version: {latest_version}) with Alias '{alias}' in MLflow Registry!"
            )

            return output_onnx_path

        except Exception as e:
            logger.error("Failed to export and register combined ONNX pipeline.")
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

            # Step 4.5: Direct ONNX Graph Merge + MLflow Registration
            combined_onnx_path = self.export_combined_onnx_pipeline(
                scaler_onnx_path="models/scaler.onnx",
                model_onnx_path="models/xgb_rul_model.onnx",
                output_onnx_path="models/full_rul_pipeline.onnx",
                registered_model_name="RUL_XGBoost_Model",
                alias="Production"
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