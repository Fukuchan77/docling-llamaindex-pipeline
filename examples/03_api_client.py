"""Ingest a document into the running API, ask a question, and print grounded sources.

Usage (with the server running, e.g. ``mise run dev``)::

    uv run python examples/03_api_client.py path/to/report.pdf "営業利益率の目標は?"

Each source's bounding box is also converted from Docling's bottom-left PDF
coordinates to the top-left coordinates a browser canvas uses (spec §5.1).
``--page-height`` defaults to US Letter (792 pt); pass your page height for
other sizes (A4 is 841.89 pt).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import httpx


def to_top_left(box: dict[str, Any], page_height: float) -> dict[str, float]:
    """Convert a BOTTOMLEFT box (l, t, r, b) to TOPLEFT (left, top, width, height)."""
    return {
        "left": round(box["l"], 2),
        "top": round(page_height - box["t"], 2),
        "width": round(box["r"] - box["l"], 2),
        "height": round(box["t"] - box["b"], 2),
    }


def main() -> None:
    """Upload, query, and print the answer with its citations."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("question")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--page-height", type=float, default=792.0)
    args = parser.parse_args()

    # Docling on CPU and a local LLM can take minutes on a large file.
    with httpx.Client(base_url=args.base_url, timeout=600) as client:
        with args.file.open("rb") as handle:
            ingest = client.post("/api/v1/ingest", files={"file": (args.file.name, handle)})
        ingest.raise_for_status()
        print(f"indexed {ingest.json()['total_nodes_created']} nodes from {args.file.name}")

        query = client.post("/api/v1/query", json={"query": args.question, "top_k": args.top_k})
        query.raise_for_status()
        body = query.json()

    print(f"\nQ: {body['query']}\nA: {body['answer']}\n")
    for i, source in enumerate(body["sources"], start=1):
        section = " > ".join(source["headings"]) or "-"
        print(f"[{i}] {source['file_name']}  score={source['score']:.3f}  section={section}")
        for box in source["bounding_boxes"]:
            print(f"    page {box['page_no']}: {to_top_left(box, args.page_height)}")


if __name__ == "__main__":
    main()
