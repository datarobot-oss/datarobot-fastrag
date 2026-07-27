# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings


class DeviceForNNComputation(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class DeviceForNNComputationSetting(BaseSettings):
    device: DeviceForNNComputation = Field(alias="DEVICE_FOR_NEURAL_NETWORK_COMPUTATIONS")
