import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_text  # noqa: E402


def test_empty():
    assert chunk_text("   ") == []


def test_chunks_respect_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, size=500, overlap=50)
    assert chunks and all(len(c) <= 500 for c in chunks)


def test_long_block_is_split():
    chunks = chunk_text("lorem " * 500, size=300, overlap=50)
    assert len(chunks) > 1 and all(len(c) <= 300 for c in chunks)


def test_sample_resume_keeps_content():
    text = (Path(__file__).resolve().parent.parent / "data" / "sample_resume.txt").read_text()
    joined = " ".join(chunk_text(text, 600, 100))
    assert "Power BI" in joined and "Sample Analytics" in joined
