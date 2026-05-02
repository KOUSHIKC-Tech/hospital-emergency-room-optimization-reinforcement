from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class Patient(BaseModel):
    id: int = Field(..., ge=0, description="Stable patient identifier.")
    severity: int = Field(..., ge=1, le=10, description="Urgency level from 1 to 10.")
    waiting_time: int = Field(0, ge=0, description="Number of steps the patient has waited.")
    condition: int = Field(
        ...,
        ge=0,
        le=10,
        description="Current health level where 0 is critical and 10 is healthy.",
    )


class Observation(BaseModel):
    patients: List[Patient]
    available_doctors: int = Field(..., ge=0)
    available_beds: int = Field(..., ge=0)
    step_count: int = Field(..., ge=0)


class Action(BaseModel):
    action_type: Literal["treat", "wait"]
    patient_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="Stable patient identifier. Preferred over patient_index.",
    )
    patient_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Legacy list position for compatibility with older callers.",
    )

    @validator("patient_index", always=True)
    def validate_target(cls, value, values):
        action_type = values.get("action_type")
        patient_id = values.get("patient_id")

        if action_type == "treat" and patient_id is None and value is None:
            raise ValueError(
                "patient_id or patient_index is required when action_type is 'treat'."
            )

        if action_type == "wait" and (patient_id is not None or value is not None):
            raise ValueError(
                "patient_id and patient_index must be omitted when action_type is 'wait'."
            )

        return value


class Reward(BaseModel):
    score: float
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Named reward components for debugging and analysis.",
    )
