"""schema_rag - a custom *schema RAG* pipeline for text-to-SQL over a 20-table SQLite DB.

Pipeline:
    question
      -> embed question            (embedder.py  - ibm-granite/granite-embedding-311m-multilingual-r2)
      -> vector search schema      (vectorstore.py)
      -> top-k candidate tables    (retriever.py)
      -> expand via FK graph       (fk_graph.py)   <- joins come from REAL db relationships, not vectors
      -> build small schema pack   (retriever.py)
      -> SQL LLM generates query   (sql_llm.py)    <- optional / pluggable
      -> validate names + run      (validator.py / pipeline.py)
"""

__version__ = "0.1.0"
