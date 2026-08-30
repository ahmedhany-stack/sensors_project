from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class SensorInput(BaseModel):
    """Schema for a single row of engine sensor measurements."""
    unit_number: int = Field(..., description="Engine ID", example=1)
    time_in_cycles: int = Field(..., description="Operating cycle number", example=1)
    
    # Operational Settings
    setting_1: float = Field(..., description="Operational Setting 1", example=-0.0007)
    setting_2: float = Field(..., description="Operational Setting 2", example=-0.0004)
    setting_3: float = Field(..., description="Operational Setting 3", example=100.0)
    
    # Sensor Measurements
    s_1: float = Field(..., example=518.67)
    s_2: float = Field(..., example=641.82)
    s_3: float = Field(..., example=1589.70)
    s_4: float = Field(..., example=1400.60)
    s_5: float = Field(..., example=14.62)
    s_6: float = Field(..., example=21.61)
    s_7: float = Field(..., example=554.36)
    s_8: float = Field(..., example=2388.06)
    s_9: float = Field(..., example=9046.19)
    s_10: float = Field(..., example=1.30)
    s_11: float = Field(..., example=47.47)
    s_12: float = Field(..., example=521.66)
    s_13: float = Field(..., example=2388.02)
    s_14: float = Field(..., example=8138.62)
    s_15: float = Field(..., example=8.4195)
    s_16: float = Field(..., example=0.03)
    s_17: float = Field(..., example=392.0)
    s_18: float = Field(..., example=2388.0)
    s_19: float = Field(..., example=100.0)
    s_20: float = Field(..., example=39.06)
    s_21: float = Field(..., example=23.4190)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "unit_number": 1,
                "time_in_cycles": 1,
                "setting_1": -0.0007,
                "setting_2": -0.0004,
                "setting_3": 100.0,
                "s_1": 518.67, "s_2": 641.82, "s_3": 1589.70, "s_4": 1400.60,
                "s_5": 14.62, "s_6": 21.61, "s_7": 554.36, "s_8": 2388.06,
                "s_9": 9046.19, "s_10": 1.30, "s_11": 47.47, "s_12": 521.66,
                "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195, "s_16": 0.03,
                "s_17": 392.0, "s_18": 2388.0, "s_19": 100.0, "s_20": 39.06, "s_21": 23.4190
            }
        }
    )


class SinglePredictionOutput(BaseModel):
    """Schema for individual engine cycle prediction output."""
    unit_number: int = Field(..., example=1)
    time_in_cycles: int = Field(..., example=1)
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life (RUL)", example=132.5)


class BatchPredictionInput(BaseModel):
    """Schema for receiving multiple sensor measurement records in a single JSON payload."""
    records: List[SensorInput]


class PredictionResponse(BaseModel):
    """Schema for the complete API response containing prediction results."""
    status: str = Field(..., example="success")
    total_records: int = Field(..., example=1)
    predictions: List[SinglePredictionOutput]


class HealthCheckResponse(BaseModel):
    """Schema for API health check and status verification."""
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    version: str = Field(..., example="1.0.0")