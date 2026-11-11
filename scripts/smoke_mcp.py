"""Start the MCP server as a subprocess over stdio, list its tools and call each one, as a real MCP client would.

Needs the index in data/ (see README). The first search loads the models, so expect it to take a while on CPU.

    python scripts/smoke_mcp.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

from mcp import StdioServerParameters
from mcp.client import Client

ROOT = Path(__file__).resolve().parents[1]


def payload(result):
    """Pull the JSON out of a tool result, whichever field the SDK filled in."""
    data = getattr(result, "structured_content", None)
    if data is None:
        data = json.loads(result.content[0].text)
    return data.get("result", data) if isinstance(data, dict) and set(data) == {"result"} else data


async def main() -> int:
    # start the server by file path from an unrelated working directory, the way a client with no cwd setting would
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "rag" / "mcp_server.py")], cwd=tempfile.gettempdir())
    async with Client(params) as client:
        tools = await client.list_tools()
        names = sorted(t.name for t in tools.tools)
        print("tools:", names)
        assert names == ["get_page", "list_books", "search_books"], names

        books = payload(await client.call_tool("list_books", {}))
        print("books:", len(books), "files, first:", books[0]["source"])

        for mode in ("hybrid", "hybrid_rerank"):
            t0 = time.perf_counter()
            hits = payload(await client.call_tool(
                "search_books", {"query": "how should I raise prices without losing customers", "mode": mode, "top": 3}))
            print(f"search_books[{mode}] {time.perf_counter() - t0:.1f}s:")
            for h in hits:
                print(f"  {h['rank']}. {h['source']} p.{h['page']} score={h['score']}")
            assert len(hits) == 3

        page = payload(await client.call_tool("get_page", {"source": hits[0]["source"], "page": hits[0]["page"]}))
        print("get_page:", page["source"], "p." + str(page["page"]), "chunks:", page["chunks"], "chars:", len(page["text"]))

        bad = await client.call_tool("search_books", {"query": "x", "mode": "magic"})
        print("bad mode is an error result:", bool(getattr(bad, "is_error", False)))
        assert getattr(bad, "is_error", False)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
