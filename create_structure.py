import os
from pathlib import Path

# قائمة المجلدات والملفات المكونة للهيكل المتقدم
project_structure = [
    # GitHub Workflows
    ".github/workflows/ci.yaml",
    ".github/workflows/cd.yaml",

    # Configs
    "config/config.yaml",
    "config/model_config.yaml",
    "config/logging_config.yaml",

    # Data & Models directories
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/features/.gitkeep",
    "models/.gitkeep",

    # Notebooks
    "notebooks/01_exploratory_analysis.ipynb",

    # Src - Utils
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/exception.py",

    # Src - Config Manager
    "src/config_manager/__init__.py",
    "src/config_manager/configuration.py",

    # Src - Components
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_validation.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",

    # Src - Pipelines
    "src/pipelines/__init__.py",
    "src/pipelines/training_pipeline.py",
    "src/pipelines/inference_pipeline.py",

    # Src - API & Monitoring
    "src/api/__init__.py",
    "src/api/app.py",
    "src/api/schemas.py",
    "src/api/monitoring.py",

    # Root Level Src Init
    "src/__init__.py",

    # Tests
    "tests/unit/__init__.py",
    "tests/unit/test_data_processing.py",
    "tests/unit/test_model.py",
    "tests/integration/__init__.py",
    "tests/integration/test_api.py",

    # Root Files
    "Dockerfile",
    "docker-compose.yaml",
    "dvc.yaml",
    "params.yaml",
    "pyproject.toml",
    "requirements.txt",
    "main.py",
    "README.md"
]

def create_structure():
    for item in project_structure:
        path = Path(item)
        
        # إنشاء المجلدات إذا لم تكن موجودة
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
            
        # إنشاء الملف إذا لم يكن موجوداً
        if not path.exists():
            path.touch()
            print(f"Created: {path}")
        else:
            print(f"Already exists: {path}")

if __name__ == "__main__":
    create_structure()