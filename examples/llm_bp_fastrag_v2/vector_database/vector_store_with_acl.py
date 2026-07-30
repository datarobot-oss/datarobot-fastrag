# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import os
from abc import ABC
from abc import abstractmethod
from enum import StrEnum
from typing import Any
from typing import Callable
from typing import List
from aiohttp import ClientSession
from aiohttp.client import DEFAULT_TIMEOUT
from langchain_core.documents import Document
from pydantic import BaseModel
from constants import DATAROBOT_IDENTITY_HEADER_NAME
from vector_database.inference.entities import DEFAULT_ADD_NEIGHBOR_CHUNKS
from vector_database.inference.entities import DEFAULT_FILTER
from vector_database.inference.entities import DEFAULT_K
from vector_database.inference.entities import MetadataColumnNames
from vector_database.inference.entities import MetadataFilterOperators
from vector_database.inference.entities import RetrievalMode
from vector_database.inference.entities import SearchResult
from vector_database.inference.entities import VectorStore

class AccessType(StrEnum):
    EXCLUDE = 'exclude'
    INCLUDE = 'include'

class DatasetAccessList(ABC, BaseModel):
    dataset_version_id: str
    files: set[str]

    def __init__(self, dataset_version_id: str, files: set[str], **kwargs: Any) -> None:
        super().__init__(dataset_version_id=dataset_version_id, files=files, **kwargs)  # type: ignore[call-arg]

    @staticmethod
    def create(dataset_version_id: str, access_type: str, files: set[str]) -> 'DatasetAccessList':
        if access_type == AccessType.EXCLUDE:
            if len(files) == 0:
                return DatasetIncludeAll(dataset_version_id)
            else:
                return DatasetExclusionList(dataset_version_id, files)
        elif access_type == AccessType.INCLUDE:
            if len(files) == 0:
                return DatasetExcludeAll(dataset_version_id)
            else:
                return DatasetInclusionList(dataset_version_id, files)
        else:
            raise ValueError(f'Unknown value {access_type} for AccessibleAccessType.')

    @abstractmethod
    def is_document_allowed(self, metadata: dict) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_filtering_condition(self) -> dict:
        raise NotImplementedError

class DatasetExclusionList(DatasetAccessList):

    def __init__(self, dataset_version_id: str, files: set[str]) -> None:
        super().__init__(dataset_version_id=dataset_version_id, files=files)

    def get_filtering_condition(self) -> dict:
        return {MetadataColumnNames.source: {MetadataFilterOperators.NIN: list(self.files)}, MetadataColumnNames.dataset_version: self.dataset_version_id}

    def is_document_allowed(self, metadata: dict) -> bool:
        source = metadata.get(MetadataColumnNames.source)
        return source not in self.files

class DatasetInclusionList(DatasetAccessList):

    def __init__(self, dataset_version_id: str, files: set[str]) -> None:
        super().__init__(dataset_version_id=dataset_version_id, files=files)

    def get_filtering_condition(self) -> dict:
        return {MetadataColumnNames.source: {MetadataFilterOperators.IN: list(self.files)}, MetadataColumnNames.dataset_version: self.dataset_version_id}

    def is_document_allowed(self, metadata: dict) -> bool:
        source = metadata.get(MetadataColumnNames.source)
        return source in self.files

class DatasetExcludeAll(DatasetAccessList):

    def __init__(self, dataset_version_id: str):
        super().__init__(dataset_version_id, set({}))

    def get_filtering_condition(self) -> dict:
        return {MetadataColumnNames.dataset_version: {MetadataFilterOperators.NE: self.dataset_version_id}}

    def is_document_allowed(self, metadata: dict) -> bool:
        return False

class DatasetIncludeAll(DatasetAccessList):

    def __init__(self, dataset_version_id: str):
        super().__init__(dataset_version_id, set({}))

    def get_filtering_condition(self) -> dict:
        return {MetadataColumnNames.dataset_version: self.dataset_version_id}

    def is_document_allowed(self, metadata: dict) -> bool:
        return True

class AccessControlChecker:

    def __init__(self, acls: list[DatasetAccessList]):
        self.acls = {acl.dataset_version_id: acl for acl in acls}

    def _is_document_allowed(self, document: Document) -> bool:
        metadata = document.metadata
        dataset_version = metadata.get(MetadataColumnNames.dataset_version)
        if dataset_version is None:
            # this is document from legacy dataset so it is allowed
            return True
        elif dataset_version in self.acls:
            return self.acls[dataset_version].is_document_allowed(metadata)
        else:
            # this is document from dataset without ACL so it is allowed
            return True

    def filter_documents(self, documents: List[Document]) -> List[Document]:
        return [document for document in documents if self._is_document_allowed(document)]

    def get_filtering_conditions(self) -> list[dict]:
        return [acl.get_filtering_condition() for acl in self.acls.values()]

class AccessControlListLoader:
    ACL_PAGE_LIMIT = 1000

    def __init__(self, datasets_with_acl: list[tuple[str, str]], datarobot_endpoint: str, authorization_header: str, identity_token_loader: Callable | None=None):
        self.datasets_with_acl = datasets_with_acl
        self.identity_token_loader = identity_token_loader
        self.datarobot_endpoint = datarobot_endpoint
        self.authorization_header = authorization_header

    async def load_access_control_list(self) -> list[DatasetAccessList]:
        acls = []
        async with ClientSession(timeout=DEFAULT_TIMEOUT, headers=self.construct_headers()) as session:  # type: ignore[arg-type]
            for dataset_id, dataset_version_id in self.datasets_with_acl:
                access_type, files = await self._load_all_access_control(session, dataset_id, dataset_version_id)
                dataset_access_list = DatasetAccessList.create(dataset_version_id, access_type, files)
                acls.append(dataset_access_list)
        return acls

    def construct_headers(self) -> dict[str, Any]:
        headers = {'Authorization': self.authorization_header}
        if self.identity_token_loader:
            identity_token = self.identity_token_loader()
            if identity_token:
                headers.update({DATAROBOT_IDENTITY_HEADER_NAME: identity_token})
        return headers

    async def _load_all_access_control(self, session: ClientSession, dataset_id: str, dataset_version_id: str) -> tuple[str, set[str]]:
        offset = 0
        etag = None
        access_type = None
        all_files: list[str] = []
        total_count = 1
        while total_count > len(all_files):
            access_type, files, etag, total_count = await self._load_access_control(session, dataset_id, dataset_version_id, offset=offset, etag=etag)
            offset += self.ACL_PAGE_LIMIT
            all_files.extend(files)
        return (access_type, set(all_files))  # type: ignore[return-value]

    async def _load_access_control(self, session: ClientSession, dataset_id: str, dataset_version_id: str, offset: int=0, etag: str | None=None) -> tuple[str, list[str], str, int]:
        url = os.path.join(self.datarobot_endpoint, f'files/{dataset_id}/versions/{dataset_version_id}/accessibleFiles?limit={self.ACL_PAGE_LIMIT}&offset={offset}')
        if etag:
            url += f'&etag={etag}'
        response = await session.get(url)
        response_json = await response.json()
        files = response_json['data']
        access_type = response_json['accessType']
        etag = response_json['etag']
        total_count = int(response_json['totalCount'])
        return (access_type, files, etag, total_count)

class VectorStoreWithAccessControl(VectorStore):
    """VectorStore implementation that supports ACL"""

    def __init__(self, vector_store: VectorStore, datarobot_endpoint: str, authorization_header: str, datasets_with_acl: list[tuple[str, str]], datasets_without_acl: list[tuple[str, str]], has_legacy_datasets: bool=False, identity_token_loader: Callable | None=None):
        self.vector_store = vector_store
        self.datasets_without_acl = [version_id for _, version_id in datasets_without_acl]
        self.has_legacy_datasets = has_legacy_datasets
        self.access_control_loader = AccessControlListLoader(datasets_with_acl, datarobot_endpoint, authorization_header, identity_token_loader)

    async def search(self, query: str, k: int=DEFAULT_K, filter: dict[str, Any] | None=DEFAULT_FILTER, add_neighbor_chunks: bool=DEFAULT_ADD_NEIGHBOR_CHUNKS, retrieval_mode: RetrievalMode=RetrievalMode.SIMILARITY, maximal_marginal_relevance_lambda: float=0.5, **kwargs: Any) -> SearchResult:
        """Add ACL related conditions to filter and call `search` of provided VectorStore"""
        acls = await self.access_control_loader.load_access_control_list()
        acl_checker = AccessControlChecker(acls=acls)
        documents, query_embeddings = await self.vector_store.search(query, k, self._update_filter(filter, acl_checker), add_neighbor_chunks, retrieval_mode, maximal_marginal_relevance_lambda, **kwargs)
        return (acl_checker.filter_documents(documents), query_embeddings)

    async def add_neighbor_chunks(self, docs: list[Document], filter: dict[str, Any] | None=DEFAULT_FILTER, apply_access_control_list: bool=False, **kwargs: Any) -> list[Document]:
        """
        Add ACL related conditions to filter
        and call `add_neighbor_chunks` of provided VectorStore
        """
        if apply_access_control_list:
            acls = await self.access_control_loader.load_access_control_list()
            acl_checker = AccessControlChecker(acls=acls)
            documents = await self.vector_store.add_neighbor_chunks(docs, self._update_filter(filter, acl_checker), **kwargs)
            return acl_checker.filter_documents(documents)
        else:
            return await self.vector_store.add_neighbor_chunks(docs, filter, **kwargs)

    def _filter_for_legacy_dataset(self) -> dict[str, dict[str, Any]]:
        return {MetadataColumnNames.dataset_version: {MetadataFilterOperators.EQ: None}}

    def _filter_without_acl(self) -> dict[str, dict[str, list]]:
        return {MetadataColumnNames.dataset_version: {MetadataFilterOperators.IN: self.datasets_without_acl}}

    def _update_filter(self, existing_filter: dict[str, Any] | None, acl_checker: AccessControlChecker) -> dict[str, Any]:
        # filtering by source always be present because we always have datasets with ACL
        or_conditions = acl_checker.get_filtering_conditions()
        if self.has_legacy_datasets or self.datasets_without_acl:
            # if any of other datasets is legacy or dataset without ACL
            # the condition that we need to add to filter is `$or`
            if self.datasets_without_acl:
                or_conditions.append(self._filter_without_acl())
            if self.has_legacy_datasets:
                or_conditions.append(self._filter_for_legacy_dataset())
        if len(or_conditions) == 1:
            new_filter = or_conditions[0]
        else:
            new_filter = {MetadataFilterOperators.OR: or_conditions}  # type: ignore[dict-item]
        if existing_filter:
            # rewrite existing filter into $and clause and add new condition
            return {MetadataFilterOperators.AND: [existing_filter, new_filter]}
        else:
            return new_filter