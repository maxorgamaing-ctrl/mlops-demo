from pydantic import BaseModel, Field
from typing import Literal


class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, description="Number of months the customer has been with the company")
    monthly_charges: float = Field(..., gt=0)
    total_charges: float = Field(..., ge=0)
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    phone_service: Literal["Yes", "No"] = "Yes"
    multiple_lines: Literal["Yes", "No", "No phone service"] = "No"
    online_security: Literal["Yes", "No", "No internet service"] = "No"
    online_backup: Literal["Yes", "No", "No internet service"] = "No"
    device_protection: Literal["Yes", "No", "No internet service"] = "No"
    tech_support: Literal["Yes", "No", "No internet service"] = "No"
    streaming_tv: Literal["Yes", "No", "No internet service"] = "No"
    streaming_movies: Literal["Yes", "No", "No internet service"] = "No"
    paperless_billing: Literal["Yes", "No"] = "Yes"
    gender: Literal["Male", "Female"] = "Male"
    senior_citizen: int = Field(default=0, ge=0, le=1)
    partner: Literal["Yes", "No"] = "No"
    dependents: Literal["Yes", "No"] = "No"


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    prediction: Literal["churn", "stay"]
    risk_tier: Literal["high", "medium", "low"]


class HealthResponse(BaseModel):
    status: str
    model_version: str
