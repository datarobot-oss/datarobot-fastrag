from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class RuntimeParameterTypes(str, Enum):
    STRING = "string"
    BOOLEAN = "boolean"
    CREDENTIAL = "credential"
    DEPLOYMENT = "deployment"
    NUMERIC = "numeric"


class RuntimeParameterPayload(BaseModel):
    type: RuntimeParameterTypes
    payload: Any


class RuntimeParameterDefinition(BaseModel):
    name: str = Field(alias="fieldName")
    type: RuntimeParameterTypes
    allow_empty: bool = Field(default=True, alias="allowEmpty")
    default: Any = Field(default=None, alias="defaultValue")
    min_value: Optional[float] = Field(default=None, alias="minValue")
    max_value: Optional[float] = Field(default=None, alias="maxValue")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class RuntimeParameterCredentialPayload(RuntimeParameterPayload):
    type: RuntimeParameterTypes = RuntimeParameterTypes.CREDENTIAL
    payload: Optional[Dict[str, Any]] = None


class RuntimeParameterStringPayload(RuntimeParameterPayload):
    type: RuntimeParameterTypes = RuntimeParameterTypes.STRING
    payload: Optional[str] = None


class RuntimeParameterBooleanPayload(RuntimeParameterPayload):
    type: RuntimeParameterTypes = RuntimeParameterTypes.BOOLEAN
    payload: Optional[bool] = None


class RuntimeParameterNumericPayload(RuntimeParameterPayload):
    type: RuntimeParameterTypes = RuntimeParameterTypes.NUMERIC
    payload: Optional[float] = None


class RuntimeParameterDeploymentPayload(RuntimeParameterPayload):
    type: RuntimeParameterTypes = RuntimeParameterTypes.DEPLOYMENT
    payload: Optional[str] = None
