"""MCP server that lets an AI agent search the indexed books.

Run it over stdio (what Claude Code and other MCP clients expect):

    python -m rag.mcp_server

It needs the index built first (`python rag/ingest.py <books dir>`, then one search to build the embeddings).
The models load on the first search call, not at start-up, so listing tools is instant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:             # lets an MCP client start this file from any working directory
    sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from rag import tools  # noqa: E402

CHUNKS = ROOT / "data" / "chunks.jsonl"
EMBEDDINGS = ROOT / "data" / "embeddings.npy"
STATS = ROOT / "data" / "ingest_stats.json"

mcp = MCPServer(
    "entrepreneurship-rag",
    instructions=(
        "Search a private collection of business books. Use search_books to find passages, then get_page to read the "
        "whole page around a hit. Always cite the source file and page number you used."
    ),
)

_retriever = None


def _get_retriever():
    """Load the models once, on first use."""
    global _retriever
    if _retriever is None:
        if not CHUNKS.exists():
            raise ToolError("index not found: run `python rag/ingest.py <books dir>` first")
        from rag.search import Retriever
        _retriever = Retriever(chunks_path=str(CHUNKS), emb_path=str(EMBEDDINGS))
    return _retriever


@mcp.tool()
def search_books(query: str, mode: str = "hybrid_rerank", top: int = 5, max_chars: int = 600) -> list[dict]:
    """Search the books for passages that answer a question.

    mode: dense, bm25, hybrid, or hybrid_rerank (default; best quality, slower on CPU).
    top: how many passages to return (1 to 20).
    max_chars: cut each passage to this many characters (0 for no limit).
    Each result has the source file, the page (or section number for EPUB files), a score and the passage text.
    The first call loads the models and can take a minute on CPU.
    """
    try:
        return tools.search_books(_get_retriever(), query, mode=mode, top=top, max_chars=max_chars)
    except ValueError as e:                  # bad arguments: report to the agent without a server traceback
        raise ToolError(str(e)) from e


@mcp.tool()
def get_page(source: str, page: int, max_chars: int = 4000) -> dict:
    """Read the full text of one page (or EPUB section) of a book, given the source and page from search_books."""
    try:
        return tools.get_page(_get_retriever().chunks, source, page, max_chars=max_chars)
    except ValueError as e:
        raise ToolError(str(e)) from e


@mcp.tool()
def list_books() -> list[dict]:
    """List the indexed books with their page counts and how many pages had no extractable text."""
    if not STATS.exists():
        raise ToolError("index not found: run `python rag/ingest.py <books dir>` first")
    return tools.list_books(json.loads(STATS.read_text(encoding="utf-8")))


def main() -> None:
    mcp.run()          # stdio transport


if __name__ == "__main__":
    main()
