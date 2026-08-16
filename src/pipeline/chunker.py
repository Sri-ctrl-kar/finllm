"""
chunker.py
------------------------------------------------------------
Splits a document's raw text into overlapping chunks, each tagged
with metadata (company, filing type, section, chunk index) so that
later, when the RAG system answers a question, it can cite exactly
which chunk of which document the answer came from.

WHY OVERLAP: if we cut chunks with no overlap, a sentence that
spans a chunk boundary gets torn in half and neither half has full
context. Overlapping by a fixed number of words means any passage
that would've been split now appears whole in at least one chunk.

WHY WORD-BASED (not character-based) sizing: LLMs and embedding
models think in tokens, which roughly track words, not characters.
Word-based chunking gives a much more predictable chunk size in
tokens than character-based chunking would.
------------------------------------------------------------
"""

from dataclasses import dataclass, field
from typing import List, Optional
import re


@dataclass
class Chunk:
    text: str
    company: str
    filing_type: str          # e.g. "10-K", "10-Q"
    source_id: str            # e.g. accession number or file name
    chunk_index: int
    section: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Human-readable citation string shown alongside any answer that uses this chunk."""
        section_part = f", {self.section}" if self.section else ""
        return f"{self.company} {self.filing_type}{section_part} (chunk #{self.chunk_index}, source: {self.source_id})"


def clean_text(raw: str) -> str:
    """Collapse whitespace/newlines from messy filing text into clean paragraphs."""
    text = re.sub(r"\r\n?", "\n", raw)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    company: str,
    filing_type: str,
    source_id: str,
    chunk_size_words: int = 250,
    overlap_words: int = 50,
    section: Optional[str] = None,
) -> List[Chunk]:
    """
    Split `text` into overlapping word-count-based chunks.

    Example: chunk_size_words=250, overlap_words=50 means each chunk
    is ~250 words, and each new chunk starts 200 words after the
    previous one started (250 - 50 overlap) — so the last 50 words
    of chunk N are also the first 50 words of chunk N+1.
    """
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    cleaned = clean_text(text)
    words = cleaned.split(" ")

    chunks: List[Chunk] = []
    step = chunk_size_words - overlap_words
    idx = 0
    chunk_index = 0

    while idx < len(words):
        window = words[idx: idx + chunk_size_words]
        if not window:
            break
        chunk_str = " ".join(window).strip()
        if chunk_str:
            chunks.append(
                Chunk(
                    text=chunk_str,
                    company=company,
                    filing_type=filing_type,
                    source_id=source_id,
                    chunk_index=chunk_index,
                    section=section,
                )
            )
            chunk_index += 1
        idx += step

    return chunks


def chunk_sections(sections: dict, company: str, filing_type: str, source_id: str, **kwargs) -> List[Chunk]:
    """
    Convenience wrapper: given a dict of {section_name: section_text}
    (e.g. {"Item 1A Risk Factors": "...", "Item 7 MD&A": "..."}),
    chunk each section separately so section boundaries are respected
    and every chunk's citation includes which section it came from.
    """
    all_chunks: List[Chunk] = []
    for section_name, section_text in sections.items():
        all_chunks.extend(
            chunk_text(section_text, company, filing_type, source_id, section=section_name, **kwargs)
        )
    return all_chunks
