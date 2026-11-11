"""Text preprocessing and quality filtering for Steam reviews."""

from __future__ import annotations

import html
import re


def fix_mojibake(text: str) -> str:
    """Memperbaiki teks mojibake UTF-8 yang ter-decode salah sebagai cp1252/latin-1."""
    if any(marker in text for marker in ("â€", "Ã", "Â")):
        for encoding in ("cp1252", "latin-1"):
            try:
                return text.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    return text


def clean_review_text(text: str) -> str:
    """Membersihkan teks ulasan Steam dari BBCode, entitas HTML, mojibake, dan whitespace berlebih.

    Args:
        text: Teks mentah ulasan Steam.

    Returns:
        Teks yang sudah dinormalisasi dan dibersihkan.
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1. Unescape entitas HTML (&amp;, &quot;, &#39;, &lt;, &gt;, dll)
    text = html.unescape(text)

    # 2. Hapus semua tag BBCode Steam ([b], [/b], [url=...], [quote], [*], [h1], dll)
    # Tag wajib diawali huruf atau asteris agar tidak menghapus angka kurung seperti [1] atau [2024]
    text = re.sub(r"\[/?(?:[a-zA-Z][a-zA-Z0-9_]*|\*)(?:=[^\]]*)?\]", "", text)

    # 3. Perbaiki karakter mojibake UTF-8 jika ada
    text = fix_mojibake(text)

    # 4. Normalisasi whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Ganti tab dan spasi berulang horizontal dengan spasi tunggal
    text = re.sub(r"[^\S\n]+", " ", text)
    # Hapus spasi di sekitar baris baru
    text = re.sub(r" *\n *", "\n", text)
    # Rapikan multi-newline (\n\n+ menjadi \n)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def is_valid_review(text: str) -> bool:
    """Memfilter ulasan sampah berdasarkan panjang kata dan rasio karakter alfabetis.

    Kriteria:
    - Menolak ulasan dengan panjang kata < 3 kata.
    - Menolak seni ASCII art (rasio karakter alfabetis < 0.20).

    Args:
        text: Teks ulasan yang sudah dibersihkan.

    Returns:
        True jika ulasan valid dan memuat konteks yang cukup, False sebaliknya.
    """
    if not isinstance(text, str):
        return False

    cleaned = text.strip()
    words = cleaned.split()
    if len(words) < 3:
        return False

    total_chars = len(cleaned)
    if total_chars == 0:
        return False

    alpha_count = sum(1 for char in cleaned if char.isalpha())
    alpha_ratio = alpha_count / total_chars

    if alpha_ratio < 0.20:
        return False

    return True
