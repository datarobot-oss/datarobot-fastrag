# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
# flake8: noqa: E501
# This template is adopted from langchain's ConversationalRetriever
CONVERSATIONAL_RETRIEVER_PROMPT = '\nGiven a chat history and the latest user query which might reference context in the chat history,\nformulate a standalone query which can be understood and be answered without the chat history.\nThis means the standalone query must contain all necessary context for an LLM to be able to answer it.\nDo NOT answer the query, just reformulate it if needed and otherwise return it as is.\n\nCHAT HISTORY: {context}\nUSER QUERY: {query}\nSTANDALONE QUERY:\n'
MULTI_STEP_RETRIEVER_PROMPT = '\nUSER QUERY: "{query}"\n\nSEARCH RESULTS: These are the results of searching for the user\'s query\n{context}\n\nSYSTEM:\nUsing the above SEARCH RESULTS and the most recent element of USER QUERY,\nwrite a numbered list of up to five new search queries to look up additional information\nrelevant to answering the user\'s query in a vector database.\nFor best results, queries should mimic the expected text content of the records\nto be retrieved, rather than being phrased like a traditional search engine query.\nFor example create two general searches relevant to the whole query,\nand three searches relevant to the three most important topics found in the\nSEARCH RESULTS.\n\nEXAMPLE ANSWER:\n1. <general search query 1>\n2. <general search query 2>\n3. <search query for one topic in the SEARCH RESULTS>\n4. <search query for another topic in the SEARCH RESULTS>\n5. <search query for a final topic in the SEARCH RESULTS>\n'