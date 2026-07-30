# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from enum import StrEnum

class EmbeddingStage(StrEnum):
    indexing = 'indexing'
    prompting = 'prompting'