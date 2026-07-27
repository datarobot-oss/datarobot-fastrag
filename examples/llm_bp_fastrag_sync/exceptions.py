# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.


class RetryableCustomModelError(Exception):
    """Raised when the custom model prediction returns one of 502, 503, 504 statuses."""


class CustomModelVectorDatabaseError(Exception):
    """Generic error for VDB custom model."""
