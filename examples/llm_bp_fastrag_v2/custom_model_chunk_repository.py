# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import os
import pickle
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from typing import Iterator
import aiofiles
import pandas as pd
from aiofiles.threadpool.binary import AsyncFileIO
from sqlalchemy import Connection
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import create_engine
from custom_model_enum import VectorDatabaseCustomModelObjects
from vector_database.inference.base_chunk_repository import BaseChunkRepository
from vector_database.inference.base_chunk_repository import ChunksCompressionType
from vector_database.inference.base_chunk_repository import DataFrameBaseChunkRepository
from vector_database.inference.base_chunk_repository import SQLBaseChunkRepository
from vector_database.inference.entities import SQLITE_METADATA_TABLE

class FileSystemChunkRepositoryMixin(BaseChunkRepository):

    @asynccontextmanager
    async def text_chunks_file(self) -> AsyncIterator[AsyncFileIO]:
        file_name = VectorDatabaseCustomModelObjects.TEXT_CHUNKS
        if await self.chunk_compression_type() == ChunksCompressionType.ZLIB:
            file_name = VectorDatabaseCustomModelObjects.TEXT_CHUNKS_ZLIB
        async with aiofiles.open(file_name, 'rb', buffering=0) as fp:
            yield fp

    async def chunk_compression_type(self) -> ChunksCompressionType:
        if self._chunk_compression_type is None:
            compression_type = ChunksCompressionType.NO_COMPRESSION
            if os.path.exists(VectorDatabaseCustomModelObjects.TEXT_CHUNKS_ZLIB):
                compression_type = ChunksCompressionType.ZLIB
            self._chunk_compression_type = compression_type
        return self._chunk_compression_type

class DataFrameCustomModelChunkRepository(FileSystemChunkRepositoryMixin, DataFrameBaseChunkRepository):

    def __init__(self) -> None:
        super().__init__()

    async def metadata_dataframe(self) -> pd.DataFrame:
        if self._metadata_dataframe is None:
            async with aiofiles.open(VectorDatabaseCustomModelObjects.TEXT_METADATA, 'rb') as file:
                self._metadata_dataframe = pickle.loads(await file.read())
        return self._metadata_dataframe

class SQLCustomModelChunkRepository(FileSystemChunkRepositoryMixin, SQLBaseChunkRepository):

    def __init__(self) -> None:
        super().__init__()
        db_file = os.path.abspath(VectorDatabaseCustomModelObjects.TEXT_METADATA_SQLITE)
        self._sql_engine = create_engine(f'sqlite+pysqlite:///{db_file}')
        self._sql_table = Table(SQLITE_METADATA_TABLE, MetaData(), autoload_with=self._sql_engine)

    @contextmanager
    def sql_connection(self) -> Iterator[Connection]:
        with self._sql_engine.connect() as connection:
            yield connection

def get_custom_model_chunk_repository() -> FileSystemChunkRepositoryMixin:
    if os.path.exists(VectorDatabaseCustomModelObjects.TEXT_METADATA_SQLITE):
        return SQLCustomModelChunkRepository()
    return DataFrameCustomModelChunkRepository()