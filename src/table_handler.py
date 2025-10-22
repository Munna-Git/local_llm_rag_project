"""
Table extraction and handling for structured data in RAG.
Ensures tables are preserved and queryable.
"""

import logging
from typing import Any, Dict, List

from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class TableHandler:
    """Handles table extraction, formatting, and context generation."""

    def __init__(self):
        logger.info("TableHandler initialized")

    def format_table_for_llm(self, table_data: Dict[str, Any]) -> str:
        """
        Format table in a way that's optimized for LLM understanding.

        Args:
            table_data: Table data from DocumentParser

        Returns:
            Formatted table string
        """
        # Prefer markdown format for better LLM comprehension
        if table_data.get("markdown"):
            formatted = table_data["markdown"]
        else:
            formatted = table_data.get("text", "")

        # Add context header
        header = f"[TABLE {table_data.get('table_id', 'unknown')}]"
        if table_data.get("page_number"):
            header += f" (Page {table_data['page_number']})"

        return f"{header}\n{formatted}"

    def create_table_context(
        self, table_data: Dict[str, Any], surrounding_text: List[str] = None
    ) -> str:
        """
        Create rich context for a table by including surrounding text.

        Args:
            table_data: Table data
            surrounding_text: Text before/after the table for context

        Returns:
            Table with context
        """
        context_parts = []

        # Add preceding context
        if surrounding_text and len(surrounding_text) > 0:
            context_parts.append("Context before table:")
            context_parts.append(surrounding_text[0][:200])  # First 200 chars

        # Add table
        context_parts.append(self.format_table_for_llm(table_data))

        # Add following context
        if surrounding_text and len(surrounding_text) > 1:
            context_parts.append("Context after table:")
            context_parts.append(surrounding_text[1][:200])  # First 200 chars

        return "\n\n".join(context_parts)

    def extract_table_with_context(
        self, elements: List[Dict[str, Any]], table_element: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract table along with surrounding context from document elements.

        Args:
            elements: All document elements
            table_element: The specific table element

        Returns:
            Table data with context
        """
        table_idx = elements.index(table_element)

        # Get 1-2 elements before and after for context
        context_before = []
        context_after = []

        # Look back for context
        for i in range(max(0, table_idx - 2), table_idx):
            if elements[i].get("text"):
                context_before.append(elements[i]["text"])

        # Look forward for context
        for i in range(table_idx + 1, min(len(elements), table_idx + 3)):
            if elements[i].get("text"):
                context_after.append(elements[i]["text"])

        return {
            "table": table_element,
            "context_before": " ".join(context_before),
            "context_after": " ".join(context_after),
        }

    def is_table_element(self, element: Dict[str, Any]) -> bool:
        """Check if an element is a table."""
        return element.get("type") == "Table" or "table_id" in element


def enhance_chunks_with_table_context(
    chunks: List[Dict[str, Any]], tables: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enhance chunks that reference tables with full table context.

    Args:
        chunks: List of document chunks
        tables: List of extracted tables

    Returns:
        Enhanced chunks with table context
    """
    table_handler = TableHandler()
    enhanced_chunks = []

    for chunk in chunks:
        # Check if chunk contains a table reference
        if chunk.get("metadata", {}).get("content_type") == "table":
            table_id = chunk["metadata"].get("table_id")

            # Find the corresponding table
            table_data = next((t for t in tables if t.get("table_id") == table_id), None)

            if table_data:
                # Enhance with formatted table
                chunk["text"] = table_handler.format_table_for_llm(table_data)
                chunk["metadata"]["enhanced"] = True

        enhanced_chunks.append(chunk)

    logger.info(f"Enhanced {len([c for c in enhanced_chunks if c.get('metadata', {}).get('enhanced')])} table chunks")
    return enhanced_chunks