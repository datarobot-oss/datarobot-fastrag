# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import asyncio
import sqlite3
import zlib
from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from enum import StrEnum
from gettext import gettext
from typing import Any
from typing import Generator
from typing import Iterable
from typing import Iterator
from typing import Sequence

import numpy as np
import pandas as pd
import sqlalchemy
from aiofiles.threadpool.binary import AsyncFileIO
from langchain.schema import Document
from loguru import logger
from sqlalchemy import Connection
from sqlalchemy import Table
from sqlalchemy import select
from vector_database.inference.entities import MetadataColumnNames
from vector_database.inference.entities import MetadataFilterOperators


def get_sqlite_max_variable_limit() -> int:
    """Get the SQLITE_MAX_VARIABLE_NUMBER variable SQLite was compiled with."""
    with sqlite3.connect(":memory:") as connection:
        limit = connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    return limit


SQLITE_MAX_VARIABLE_NUMBER = get_sqlite_max_variable_limit()  # For our docker images this is 999


class ChunksCompressionType(StrEnum):
    """Supported types of text chunks compression."""

    # No compression for VDBs created with older versions of buzok
    NO_COMPRESSION = "no_compression"
    # Zlib compression without dictionary
    ZLIB = "zlib"


def cast_numpy_to_native(value: Any) -> Any:
    """Cast numpy types to native Python types."""
    if isinstance(value, np.integer | np.floating):
        return value.item()
    elif isinstance(value, dict):
        return {k: cast_numpy_to_native(v) for k, v in value.items()}
    return value


def decompress_chunk(data: bytes, compression_type: ChunksCompressionType) -> bytes:
    # We implemented chunk compression as part of 10.2 release.
    # While the compression function always uses the latest and the best available algorithm,
    # the decompression function must be backward compatible with older versions of buzok
    # and handle all historically used compression algorithms, as well as no compression.
    if compression_type == ChunksCompressionType.NO_COMPRESSION:
        return data
    if compression_type == ChunksCompressionType.ZLIB:
        decompressor = zlib.decompressobj()
        result = decompressor.decompress(data)
        result += decompressor.flush()
        return result
    raise NotImplementedError(f"Chunk compression type {compression_type} not implemented")


class BaseChunkRepository(ABC):
    RESERVED_METADATA_FIELDS = {MetadataColumnNames.chunk_size.value}

    def __init__(self, **args: Any) -> None:
        self._offsets: np.ndarray | None = None
        self._chunk_compression_type: ChunksCompressionType | None = None
        self.logger = logger

    async def offsets(self) -> np.ndarray:
        if self._offsets is None:
            # convert array with size of chunks to array with offsets - e.g. content
            # of chunk N can be found in kb_file between offset_array[N] and offset_array[N+1]
            # dtype=np.int64 type is explicitly forced here instead of the unsigned uint64.
            # Offsets for byte ranges are always positive. Using unsigned int here in can
            # lead to silent float64 conversion (https://github.com/numpy/numpy/issues/5745)
            # This can lead to selective, hard to debug, data retrieval errors.
            self._offsets = np.insert(np.cumsum(await self.chunk_size(), dtype=np.int64), 0, 0)
        return self._offsets

    @abstractmethod
    async def chunk_size(self) -> np.ndarray:
        pass

    @abstractmethod
    @asynccontextmanager
    async def text_chunks_file(self) -> AsyncIterator[AsyncFileIO]:
        yield AsyncFileIO("", None, None)
        raise NotImplementedError()

    @abstractmethod
    async def chunk_compression_type(self) -> ChunksCompressionType:
        raise NotImplementedError()

    @abstractmethod
    async def _get_metadata_for_indices(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        pass

    def _get_metadata_columns_to_include(self, available_columns: Iterable[str]) -> list[str]:
        metadata_columns = [
            column_name
            for column_name in available_columns
            if column_name not in self.RESERVED_METADATA_FIELDS
        ]
        return metadata_columns

    async def retrieve_text_chunks(self, indices: Sequence[int]) -> list[Document]:
        offsets = await self.offsets()
        metadata_for_indices = await self._get_metadata_for_indices(indices)
        documents = []
        self.logger.debug("Retrieving text chunks and preparing Document objects to return")
        async with self.text_chunks_file() as text_file:
            for index, metadata_for_index in zip(indices, metadata_for_indices):
                await text_file.seek(offsets[index])
                chunk_bytes = await text_file.read(offsets[index + 1] - offsets[index])
                chunk_compression_type = await self.chunk_compression_type()
                chunk_bytes = await asyncio.to_thread(
                    decompress_chunk, chunk_bytes, chunk_compression_type
                )
                documents.append(
                    Document(page_content=chunk_bytes.decode("utf-8"), metadata=metadata_for_index)
                )
        self.logger.debug(
            "Finished retrieving text chunks and preparing Document objects to return"
        )
        return documents

    @abstractmethod
    async def get_indices_matching_filters(self, filters: dict[str, Any] | None) -> np.ndarray:
        pass


class DataFrameBaseChunkRepository(BaseChunkRepository):
    def __init__(self, **args: Any) -> None:
        super().__init__(**args)
        self._metadata_dataframe: pd.DataFrame | None = None

    async def chunk_size(self) -> np.ndarray:
        metadata_dataframe = await self.metadata_dataframe()
        return metadata_dataframe["chunk_size"].to_numpy()

    @abstractmethod
    async def metadata_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError()

    async def _get_metadata_for_indices(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        metadata_dataframe = await self.metadata_dataframe()
        available_columns = metadata_dataframe.columns
        metadata_columns = self._get_metadata_columns_to_include(available_columns)
        metadatas = []
        for index in indices:
            metadata_column = metadata_columns
            metadata = {
                column: cast_numpy_to_native(metadata_dataframe.at[index, column])
                for column in metadata_column
            }
            metadata[MetadataColumnNames.chunk_id.value] = index
            metadatas.append(metadata)
        return metadatas

    async def get_indices_matching_filters(self, filters: dict[str, Any] | None) -> np.ndarray:
        raise NotImplementedError()


class SQLBaseChunkRepository(BaseChunkRepository):
    def __init__(self, **args: Any) -> None:
        super().__init__(**args)
        self._sql_table: Table | None = None

    @property
    def sql_table(self) -> Table:
        if self._sql_table is None:
            raise ValueError("SQL table not set")
        return self._sql_table

    @abstractmethod
    @contextmanager
    def sql_connection(self) -> Iterator[Connection]:
        pass

    async def chunk_size(self) -> np.ndarray:
        return await asyncio.to_thread(self._chunk_sizes)

    def _chunk_sizes(self) -> np.ndarray:
        self.logger.debug("Getting chunk sizes from SQL database")
        with self.sql_connection() as connection:
            s = select(self.sql_table.c[MetadataColumnNames.chunk_size.value]).order_by(
                MetadataColumnNames.chunk_id.value
            )
            chunk_sizes = connection.execute(s).fetchall()
        sizes = np.array([s[0] for s in chunk_sizes])
        self.logger.debug("Finished getting chunk sizes from SQL database")
        return sizes

    async def _get_metadata_for_indices(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.__get_metadata_for_indices, indices)

    def __get_metadata_for_indices(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        logger = self.logger.bind(num_indices=len(indices))
        logger.debug("Getting metadata for indices from SQL database")
        columns_to_include = self._get_metadata_columns_to_include(
            [c.name for c in self.sql_table.columns]
        )
        data = {}
        with self.sql_connection() as connection:
            # SQLite has a max number of variables you can include in one query. In case we have
            # more, we have to query in batches
            for inices_batch in self.batch(indices, batch_size=SQLITE_MAX_VARIABLE_NUMBER):  # type: ignore[call-overload]
                s = select(self.sql_table.c[*columns_to_include,]).where(
                    self.sql_table.c[MetadataColumnNames.chunk_id.value].in_(inices_batch)
                )
                metadata = connection.execute(s).fetchall()
                for d in metadata:
                    mapping = d._mapping
                    data[mapping[MetadataColumnNames.chunk_id.value]] = mapping
        metadata = [data[i] for i in indices if i in data]  # type: ignore
        logger.debug("Finished getting metadata for indices from SQL database")
        return metadata  # type: ignore[return-value]

    def batch(self, list_: Sequence[int], batch_size: int) -> Generator[Sequence[int], None, None]:
        length = len(list_)
        for idx in range(0, length, batch_size):
            yield list_[idx : min(idx + batch_size, length)]

    async def get_indices_matching_filters(self, filters: dict[str, Any] | None) -> np.ndarray:
        return await asyncio.to_thread(self._get_indices_matching_filters, filters)

    def _build_comparison_filter(self, column: str, operator_dict: dict) -> Any:  # noqa: PLR0911
        column_ref = self.sql_table.c[column]
        for operator, value in operator_dict.items():
            try:
                # Putting the operators into the variables so they aren't translated
                # accidentally
                match operator:
                    case MetadataFilterOperators.EQ.value:
                        return column_ref == value
                    case MetadataFilterOperators.NE.value:
                        return column_ref != value
                    case MetadataFilterOperators.GT.value:
                        return column_ref > value
                    case MetadataFilterOperators.GTE.value:
                        return column_ref >= value
                    case MetadataFilterOperators.LT.value:
                        return column_ref < value
                    case MetadataFilterOperators.LTE.value:
                        return column_ref <= value
                    case MetadataFilterOperators.IN.value:
                        return column_ref.in_(value)
                    case MetadataFilterOperators.NIN.value:
                        return ~column_ref.in_(value)
                    case MetadataFilterOperators.CONTAINS.value:
                        return column_ref.like(f"%{value}%")
                    case MetadataFilterOperators.NOT_CONTAINS.value:
                        return ~column_ref.like(f"%{value}%")
                    case _:
                        raise ValueError(
                            gettext(
                                "The metadata filter contains an unsupported operator. Supported operators are: {supported_operators}"
                            ).format(
                                supported_operators="$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $contains, $not_contains"
                            )
                        )
            except sqlalchemy.exc.ArgumentError:
                raise ValueError(
                    gettext("Invalid metadata filter. One of the values had an incompatible type.")
                )

    def _build_filter_expression(self, filters: dict | None) -> Any:
        """Build a SQLAlchemy filter expression from a filter dictionary.

        Supports:
        - None or empty dict (no filters): returns True
        - Multiple field filters (implicit AND): {"a": 1, "b": "b"}
        - Comparison operators: {"field": {"$gt": 5}}
        - Logical operators: {"$and": [...], "$or": [...]}
        - Multiple logical operators at same level: {"$and": [...], "$or": [...]}
        - Nested combinations of the above
        """
        # Handle None or empty dict case
        if not filters:
            return True
        conditions = []
        for field, value in filters.items():
            # Handle logical operators first
            if field == MetadataFilterOperators.AND.value:
                conditions.append(
                    sqlalchemy.and_(*[self._build_filter_expression(f) for f in value])
                )
            elif field == MetadataFilterOperators.OR.value:
                conditions.append(
                    sqlalchemy.or_(*[self._build_filter_expression(f) for f in value])
                )
            elif isinstance(value, dict):
                # Complex comparison with operators
                conditions.append(self._build_comparison_filter(field, value))
            else:
                # Simple equality comparison
                conditions.append(self.sql_table.c[field] == value)
        if len(conditions) == 0:
            raise ValueError(gettext("Invalid filter dictionary structure."))
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return sqlalchemy.and_(*conditions)

    def _get_indices_matching_filters(self, filters: dict[str, Any] | None) -> np.ndarray:
        self.logger.debug("Getting indices matching filters from SQL database")
        filters = {} if filters is None else filters
        with self.sql_connection() as connection:
            s = select(self.sql_table.c[MetadataColumnNames.chunk_id.value]).where(
                self._build_filter_expression(filters)
            )
            try:
                indices = connection.execute(s).fetchall()
            except sqlalchemy.exc.ProgrammingError:
                raise ValueError(gettext("Invalid filter."))
        result = np.array([i[0] for i in indices])
        self.logger.debug("Finished retrieving indices matching filters from the SQL database.")
        return result
