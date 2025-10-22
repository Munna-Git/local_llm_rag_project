"""
Advanced semantic chunking with hierarchical structure.
Replaces naive fixed-size chunking with intelligent topic-based segmentation.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Implements semantic chunking using sentence embeddings.
    Detects topic boundaries by measuring embedding similarity.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 500,
        min_chunk_size: int = 100,
    ):
        """
        Initialize semantic chunker.

        Args:
            embedding_model: SentenceTransformer model for embeddings
            similarity_threshold: Threshold for topic boundary detection (0-1)
            max_chunk_size: Maximum words per chunk
            min_chunk_size: Minimum words per chunk
        """
        self.model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        logger.info("SemanticChunker initialized with threshold={:.2f}".format(similarity_threshold))

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text using semantic similarity.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunks with metadata
        """
        metadata = metadata or {}

        # Split into sentences
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [{"text": text, "metadata": metadata}]

        # Generate embeddings for sentences
        embeddings = self.model.encode(sentences)

        # Calculate similarity between consecutive sentences
        similarities = self._calculate_similarities(embeddings)

        # Find topic boundaries
        boundaries = self._find_boundaries(similarities)

        # Create chunks
        chunks = self._create_chunks(sentences, boundaries, metadata)

        logger.info(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics."""
        import re

        # Simple sentence splitter (can be improved with nltk/spacy)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_similarities(self, embeddings: np.ndarray) -> List[float]:
        """Calculate cosine similarity between consecutive sentence embeddings."""
        similarities = []

        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(float(sim))

        return similarities

    def _find_boundaries(self, similarities: List[float]) -> List[int]:
        """
        Find topic boundaries where similarity drops below threshold.

        Returns indices where new chunks should start.
        """
        boundaries = [0]  # Start with first sentence

        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)

        return boundaries

    def _create_chunks(
        self, sentences: List[str], boundaries: List[int], metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create chunks from sentences and boundaries."""
        chunks = []
        boundaries.append(len(sentences))  # Add end boundary

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            # Enforce size constraints
            if len(chunk_text.split()) > self.max_chunk_size:
                # Split large chunks
                sub_chunks = self._split_large_chunk(chunk_text)
                chunks.extend(sub_chunks)
            elif len(chunk_text.split()) < self.min_chunk_size and chunks:
                # Merge small chunks with previous
                chunks[-1]["text"] += " " + chunk_text
            else:
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "sentence_start": start_idx,
                            "sentence_end": end_idx,
                        },
                    }
                )

        return chunks

    def _split_large_chunk(self, text: str) -> List[Dict[str, Any]]:
        """Split chunks that exceed max_chunk_size."""
        words = text.split()
        sub_chunks = []

        for i in range(0, len(words), self.max_chunk_size):
            chunk_words = words[i : i + self.max_chunk_size]
            sub_chunks.append({"text": " ".join(chunk_words), "metadata": {}})

        return sub_chunks


class HierarchicalChunker:
    """
    Creates hierarchical chunks with parent-child relationships.
    Maintains document structure for better context.
    """

    def __init__(self, semantic_chunker: SemanticChunker):
        self.semantic_chunker = semantic_chunker
        logger.info("HierarchicalChunker initialized")

    def chunk_document(
        self, parsed_doc: Dict[str, Any], document_name: str
    ) -> List[Dict[str, Any]]:
        """
        Create hierarchical chunks from parsed document.

        Args:
            parsed_doc: Output from DocumentParser
            document_name: Name of the document

        Returns:
            List of chunks with hierarchical metadata
        """
        all_chunks = []
        elements = parsed_doc["elements"]
        tables = parsed_doc.get("tables", [])

        # Create document-level summary
        doc_text = self._create_document_summary(elements)
        doc_metadata = {
            "document_name": document_name,
            "level": "document",
            "total_elements": len(elements),
        }

        # Track current section for hierarchy
        current_section = None
        section_chunks = []

        for element in elements:
            element_type = element.get("type")
            text = element.get("text", "")

            if element_type == "Title":
                # Save previous section
                if section_chunks:
                    all_chunks.extend(self._finalize_section(current_section, section_chunks))
                    section_chunks = []

                current_section = text

            # Handle tables separately
            elif element.get("table_id"):
                table_data = self._find_table(element["table_id"], tables)
                if table_data:
                    table_chunk = self._create_table_chunk(
                        table_data, current_section, document_name
                    )
                    all_chunks.append(table_chunk)

            # Regular text elements
            elif text:
                metadata = {
                    "document_name": document_name,
                    "section": current_section,
                    "page_number": element.get("page_number"),
                    "element_type": element_type,
                }

                # Use semantic chunking for long text
                if len(text.split()) > 50:
                    chunks = self.semantic_chunker.chunk_text(text, metadata)
                    section_chunks.extend(chunks)
                else:
                    section_chunks.append({"text": text, "metadata": metadata})

        # Finalize last section
        if section_chunks:
            all_chunks.extend(self._finalize_section(current_section, section_chunks))

        logger.info(
            f"Created {len(all_chunks)} hierarchical chunks for document {document_name}"
        )
        return all_chunks

    def _create_document_summary(self, elements: List[Dict[str, Any]]) -> str:
        """Create a summary of the document from titles and first paragraphs."""
        summary_parts = []

        for element in elements[:10]:  # First 10 elements
            if element.get("type") == "Title":
                summary_parts.append(element.get("text", ""))

        return " | ".join(summary_parts)

    def _find_table(self, table_id: str, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find table data by ID."""
        for table in tables:
            if table.get("table_id") == table_id:
                return table
        return {}

    def _create_table_chunk(
        self, table_data: Dict[str, Any], section: str, document_name: str
    ) -> Dict[str, Any]:
        """Create a special chunk for tables with structured data."""
        # Use markdown representation for better LLM understanding
        table_text = f"Table from section '{section}':\n{table_data.get('markdown', table_data.get('text'))}"

        return {
            "text": table_text,
            "metadata": {
                "document_name": document_name,
                "section": section,
                "content_type": "table",
                "table_id": table_data["table_id"],
                "page_number": table_data.get("page_number"),
            },
        }

    def _finalize_section(
        self, section_name: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add section-level metadata to chunks."""
        for chunk in chunks:
            chunk["metadata"]["parent_section"] = section_name

        return chunks


def create_hierarchical_chunks(
    parsed_doc: Dict[str, Any], document_name: str, embedding_model: SentenceTransformer
) -> List[Dict[str, Any]]:
    """
    Convenience function to create hierarchical semantic chunks.

    Args:
        parsed_doc: Parsed document from DocumentParser
        document_name: Name of the document
        embedding_model: SentenceTransformer model

    Returns:
        List of hierarchical chunks ready for indexing
    """
    semantic_chunker = SemanticChunker(embedding_model)
    hierarchical_chunker = HierarchicalChunker(semantic_chunker)

    return hierarchical_chunker.chunk_document(parsed_doc, document_name)