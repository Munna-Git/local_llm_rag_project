# src\document_parser.py

"""
Advanced document parser with layout-aware extraction.
Handles tables, images, and document structure intelligently.
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Element,
    Table,
    Title,
    NarrativeText,
    ListItem,
    Image,
)

from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class DocumentParser:
    """Handles intelligent PDF parsing with layout detection."""

    def __init__(self, extract_images: bool = True, extract_tables: bool = True):
        """
        Initialize the document parser.

        Args:
            extract_images: Whether to extract image descriptions
            extract_tables: Whether to extract and structure tables
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        logger.info("DocumentParser initialized with layout-aware extraction.")

    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PDF with layout detection.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dict containing structured document data with:
                - elements: List of document elements with metadata
                - tables: List of extracted tables
                - images: List of image descriptions
                - metadata: Document-level metadata
        """
        logger.info(f"Parsing PDF with layout detection: {file_path}")

        try:
            # Use unstructured to parse with layout detection
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",  # High-resolution layout detection
                infer_table_structure=self.extract_tables,
                extract_images_in_pdf=False,  # We'll handle images separately
                include_page_breaks=True,
            )

            parsed_data = self._process_elements(elements, file_path)
            logger.info(
                f"Successfully parsed {len(parsed_data['elements'])} elements from {Path(file_path).name}"
            )

            return parsed_data

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            # Fallback to basic extraction
            return self._fallback_parsing(file_path)

    def _process_elements(
        self, elements: List[Element], file_path: str
    ) -> Dict[str, Any]:
        """Process extracted elements into structured format."""

        structured_elements = []
        tables = []
        images = []
        current_section = None

        for idx, element in enumerate(elements):
            element_data = {
                "type": type(element).__name__,
                "text": str(element),
                "page_number": getattr(element.metadata, "page_number", None),
                "element_id": f"{Path(file_path).stem}_{idx}",
            }

            # Handle different element types
            if isinstance(element, Title):
                current_section = str(element)
                element_data["is_title"] = True
                element_data["section"] = current_section

            elif isinstance(element, Table):
                # Extract table structure
                table_data = self._extract_table_structure(element, idx)
                tables.append(table_data)
                element_data["table_id"] = table_data["table_id"]
                element_data["section"] = current_section

            elif isinstance(element, (NarrativeText, ListItem)):
                element_data["section"] = current_section

            elif isinstance(element, Image):
                # Placeholder for image handling
                image_data = {
                    "image_id": f"img_{idx}",
                    "page_number": element_data["page_number"],
                    "context": current_section,
                }
                images.append(image_data)
                element_data["image_id"] = image_data["image_id"]

            structured_elements.append(element_data)

        return {
            "elements": structured_elements,
            "tables": tables,
            "images": images,
            "metadata": {
                "filename": Path(file_path).name,
                "total_elements": len(structured_elements),
                "total_tables": len(tables),
                "total_images": len(images),
            },
        }

    def _extract_table_structure(self, table_element: Table, idx: int) -> Dict[str, Any]:
        """Extract table as structured markdown."""

        try:
            # Get table as HTML, then convert to markdown for better LLM understanding
            table_html = table_element.metadata.text_as_html

            table_data = {
                "table_id": f"table_{idx}",
                "text": str(table_element),
                "html": table_html if table_html else None,
                "markdown": self._html_table_to_markdown(table_html)
                if table_html
                else str(table_element),
                "page_number": getattr(table_element.metadata, "page_number", None),
            }

            logger.debug(f"Extracted table {table_data['table_id']}")
            return table_data

        except Exception as e:
            logger.warning(f"Could not extract table structure: {e}")
            return {
                "table_id": f"table_{idx}",
                "text": str(table_element),
                "markdown": str(table_element),
                "page_number": getattr(table_element.metadata, "page_number", None),
            }

    def _html_table_to_markdown(self, html: Optional[str]) -> str:
        """Convert HTML table to markdown format."""
        if not html:
            return ""

        try:
            # Simple HTML to markdown conversion for tables
            # This is a basic implementation - could be enhanced
            import re

            # Remove HTML tags but preserve structure
            text = re.sub(r"<tr>", "\n", html)
            text = re.sub(r"<td>|<th>", "| ", text)
            text = re.sub(r"</td>|</th>", " ", text)
            text = re.sub(r"<[^>]+>", "", text)

            return text.strip()

        except Exception as e:
            logger.warning(f"Error converting HTML table to markdown: {e}")
            return html

    def _fallback_parsing(self, file_path: str) -> Dict[str, Any]:
        """Fallback to basic PyPDF2 parsing if unstructured fails."""
        logger.warning(f"Using fallback parsing for {file_path}")

        from PyPDF2 import PdfReader

        text_elements = []
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_elements.append(
                        {
                            "type": "NarrativeText",
                            "text": text,
                            "page_number": page_num + 1,
                            "element_id": f"{Path(file_path).stem}_fallback_{page_num}",
                        }
                    )
        except Exception as e:
            logger.error(f"Fallback parsing also failed: {e}")

        return {
            "elements": text_elements,
            "tables": [],
            "images": [],
            "metadata": {
                "filename": Path(file_path).name,
                "total_elements": len(text_elements),
                "total_tables": 0,
                "total_images": 0,
                "fallback_used": True,
            },
        }


def parse_document(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to parse a document.

    Args:
        file_path: Path to the document

    Returns:
        Structured document data
    """
    parser = DocumentParser()
    return parser.parse_pdf(file_path)