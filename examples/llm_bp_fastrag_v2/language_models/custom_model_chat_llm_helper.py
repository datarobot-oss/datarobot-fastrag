# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import hashlib
import os
from typing import Any
from typing import Dict
from aiohttp import ClientSession
from datarobot_dome.constants import GuardStage
from datarobot_dome.constants import GuardType
from datarobot_dome.guards.base import get_metric_column_name
from humps import decamelize
from language_models.language_model_interface import BaseExecutionStatus
from language_models.language_model_interface import BaseMetricMetadata
from language_models.language_model_interface import BasePipelineStage
DATAROBOT_DOME_STAGE_TO_BUZOK_STAGE_MAP = {GuardStage.PROMPT: BasePipelineStage.PROMPT_PIPELINE, GuardStage.RESPONSE: BasePipelineStage.RESPONSE_PIPELINE}
# The API defines also PII (deprecated) and userModel (defined in the lib as ootb type with
# OOTBType.USER_MODEL).
API_GUARD_TYPE_TO_MODERATION_LIB_GUARD_TYPE: Dict[str, GuardType] = {'guardModel': GuardType.MODEL, 'nemo': GuardType.NEMO_GUARDRAILS, 'ootb': GuardType.OOTB, 'nemoEvaluator': GuardType.NEMO_EVALUATOR}

def get_metrics_metadata(guard_configurations: list[Dict[str, Any]], moderation_lib_results: Dict[str, Any]) -> list[BaseMetricMetadata]:
    metrics: list[BaseMetricMetadata] = []
    for guard_configuration in guard_configurations:
        stages = guard_configuration['stages']
        for stage in stages:
            guard_type = API_GUARD_TYPE_TO_MODERATION_LIB_GUARD_TYPE[guard_configuration['type']]
            # The moderation lib doesn't provide metric results in some cases. Let's skip those.
            if not _does_guard_provide_metric_result(guard_type, guard_configuration):
                continue
            model_guard_target_name = None
            model_info = guard_configuration.get('model_info')
            if model_info is not None:
                model_guard_target_name = model_info.get('output_column_name')
            ootb_guard_type = guard_configuration['ootb_type']
            metric_result_key = get_metric_column_name(guard_type=guard_type, ootb_type=ootb_guard_type, stage=stage, model_guard_target_name=model_guard_target_name, nemo_evaluator_type=guard_configuration.get('nemo_evaluator_type'))
            is_metric_value_present = metric_result_key in moderation_lib_results
            execution_status = BaseExecutionStatus.COMPLETED if is_metric_value_present else BaseExecutionStatus.ERROR
            error_message = None
            if not is_metric_value_present:
                error_message = f'Metric value was expected in chat_response.datarobot_moderations.{metric_result_key} but was not found.'
            # get guard type used for generating a custom_model_guard_id
            # Note: It appears that snake case guard type should be passed into get_guard_id, but
            # we have always used guard_configuration["type"], which should be camelCase. So trying
            # to do the right thing for nemo evaluator, but keep thing unchanged for others.
            guard_type = GuardType.NEMO_EVALUATOR if guard_configuration['type'] == 'nemoEvaluator' else guard_configuration['type']
            guard_name = guard_configuration['name']
            metric = BaseMetricMetadata(name=_get_metric_name(guard_name, stage, stages), value=moderation_lib_results.get(metric_result_key), stage=DATAROBOT_DOME_STAGE_TO_BUZOK_STAGE_MAP[stage], execution_status=execution_status, error_message=error_message, custom_model_guard_id=get_guard_id(guard_name, guard_type, stage, ootb_guard_type, guard_configuration.get('nemo_evaluator_type')))
            metrics.append(metric)
    return metrics

def _get_metric_name(guard_name: str, stage: GuardStage, stages: list[GuardStage]) -> str:
    # This implementation must remain synchronized
    # with the internal logic for insight name retrieval
    if len(stages) <= 1:
        metric_name = guard_name
    else:
        metric_name = f'{guard_name} - {_get_stage_suffix(stage)}'
    return metric_name

def _get_stage_suffix(stage: GuardStage) -> str:
    # This implementation must remain synchronized
    # with the internal logic for stage suffix retrieval
    if stage == GuardStage.PROMPT:
        return 'Prompt'
    elif stage == GuardStage.RESPONSE:
        return 'Response'
    else:
        raise ValueError(f'Unknown stage: {stage}')

async def get_guard_configuration_for_custom_model(session: ClientSession, datarobot_endpoint: str, custom_model_id: str) -> list[Dict[str, Any]]:
    response = await session.get(os.path.join(datarobot_endpoint, f'customModels/{custom_model_id}/'))
    custom_model_version_id = (await response.json())['latestVersion']['id']
    params = {'entityId': str(custom_model_version_id), 'entityType': 'customModelVersion'}
    response = await session.get(os.path.join(datarobot_endpoint, 'guardConfigurations/'), params=params)
    guard_configurations = await response.json()
    return decamelize(guard_configurations['data'])

def _does_guard_provide_metric_result(guard_type: GuardType, guard_configuration: Dict[str, Any]) -> bool:
    result = True
    if guard_type == GuardType.MODEL:
        taget_type = guard_configuration.get('model_info', {}).get('target_type')
        if taget_type == 'Multiclass':
            result = False
    elif guard_type == GuardType.NEMO_GUARDRAILS:
        result = False
    return result

def get_guard_id(name: str, guard_type: str, stage: str, ootb_type: str | None=None, nemo_evaluator_type: str | None=None) -> str:
    """Use this function to create a unique ID for a guard metric. This is required as the
    evaluation dataset metric aggregation job needs a mechanism to match insights
    configurations with computed metrics.
    We can't use the guard ID as the ID of a guard will change whenever a change to any of the
    guards is done via the UI or API.

    This implementation must remain synchronized
    with the internal logic for guard id retrieval
    """
    id_string = f'{name}_{guard_type}_{stage}_{ootb_type}'
    if nemo_evaluator_type:
        id_string += f'_{nemo_evaluator_type}'
    m = hashlib.sha1()
    m.update(id_string.encode('utf-8'))
    return m.hexdigest()