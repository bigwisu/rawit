# Copyright 2026 Wisu Suntoyo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pydantic schemas for the POST /v1/systemone endpoint."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class OptionSchema(BaseModel):
    id: str = Field(..., description="Stable identifier for this option.")
    label: str = Field(..., description="Human-readable option text fed to the model.")


class SystemOneRequest(BaseModel):
    context: str = Field(..., description="State / document the decision is about.")
    question: str = Field(..., description="Rubric or classification instruction.")
    type: str = Field("choice", description="Decision primitive: choice | score | noul.")
    options: List[OptionSchema] = Field(..., min_length=2)
    escalate_threshold: float = Field(0.65, ge=0.0, le=1.0)
    max_len: Optional[int] = Field(None, ge=64, le=8192)
    model: Optional[str] = Field(None)


class SystemOneResponse(BaseModel):
    decision: str
    confidence: float
    probabilities: Dict[str, float]
    escalate: bool
    escalation_score: float
    execution_time_ms: float
    model: str
