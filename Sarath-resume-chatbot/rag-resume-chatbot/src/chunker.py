"""Paragraph-aware chunking with overlap, tuned for resumes."""
import re


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long(block: str, size: int, overlap: int) -> list[str]:
    """Sliding window over a long block, snapping to whitespace."""
    out, start, n = [], 0, len(block)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = block.rfind(" ", start + size // 2, end)
            end = cut if cut != -1 else end
        piece = block[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_text(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    text = _clean(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if len(p) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_split_long(p, size, overlap))
        elif len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks
