"""Unit tests untuk modul cleaner: pembersihan BBCode, HTML entities, dan validasi ulasan."""

import pytest

from src.ingestion.cleaner import clean_review_text, is_valid_review


def test_clean_bbcode_tags():
    """Uji penghapusan berbagai macam tag BBCode Steam."""
    raw_text = (
        "[h1]Best Indie Game Ever[/h1]\n\n"
        "[b]Hades[/b] is [i]absolutely[/i] [u]magnificent[/u]!\n"
        "[quote=Zag]Father, I will escape.[/quote]\n"
        "[url=https://store.steampowered.com/app/1145360]Check it out![/url]\n"
        "[list][*]Great combat[*]Superb soundtrack[/list]\n"
        "[spoiler]Major plot secret[/spoiler]\n"
        "[code]sudo apt install hades[/code]\n"
        "[table][tr][th]Pros[/th][/tr][tr][td]Everything[/td][/tr][/table]"
    )
    cleaned = clean_review_text(raw_text)

    # Pastikan tidak ada satupun tag BBCode tersisa
    assert "[" not in cleaned
    assert "]" not in cleaned

    # Pastikan konten teks penting di dalam tag tetap ada
    assert "Best Indie Game Ever" in cleaned
    assert "Hades is absolutely magnificent!" in cleaned
    assert "Father, I will escape." in cleaned
    assert "Check it out!" in cleaned
    assert "Great combat" in cleaned
    assert "Superb soundtrack" in cleaned
    assert "Major plot secret" in cleaned
    assert "Everything" in cleaned


def test_clean_html_entities():
    """Uji unescape entitas HTML umum."""
    raw_text = "It&#39;s a &quot;masterpiece&quot; &amp; costs &lt; $20 &gt; $10."
    cleaned = clean_review_text(raw_text)

    assert cleaned == "It's a \"masterpiece\" & costs < $20 > $10."


def test_preserve_bracketed_numbers():
    """Uji bahwa angka dalam kurung siku seperti [10/10], [2024], dan [1] tidak terhapus."""
    raw_text = "I rate this game [10/10]! Best release of [2024], truly [1] of a kind."
    cleaned = clean_review_text(raw_text)

    assert "[10/10]" in cleaned
    assert "[2024]" in cleaned
    assert "[1]" in cleaned


def test_clean_mojibake_and_whitespace():
    """Uji perbaikan mojibake dan normalisasi whitespace multi-newline serta tab."""
    raw_text = "  Itâ€™s    great\t\tgameplay!\n\n\n\nReally   enjoyed it.  \r\n"
    cleaned = clean_review_text(raw_text)

    assert "’" in cleaned or "'" in cleaned
    assert "\t" not in cleaned
    assert "    " not in cleaned
    assert "\n\n" not in cleaned
    assert cleaned.startswith("It")
    assert cleaned.endswith("Really enjoyed it.")


def test_reject_ascii_art():
    """Uji penolakan ulasan seni ASCII art (rasio alfabet < 0.20)."""
    # 1. Braille art
    braille_art = "⣿" * 40 + "\n" + "⣿" * 40 + "\n" + "⣿" * 40
    assert is_valid_review(braille_art) is False

    # 2. Block/box drawing art
    box_art = "░▒▓█░▒▓█░▒▓█░▒▓█░▒▓█\n░▒▓█░▒▓█░▒▓█░▒▓█░▒▓█"
    assert is_valid_review(box_art) is False

    # 3. Shrug emoticon spam
    shrug_spam = "¯\\_(ツ)_/¯ ¯\\_(ツ)_/¯ ¯\\_(ツ)_/¯"
    assert is_valid_review(shrug_spam) is False

    # 4. Dot art / border spam
    border_spam = "........................................\n----------------------------------------\n========================================"
    assert is_valid_review(border_spam) is False


def test_reject_short_reviews():
    """Uji penolakan ulasan terlalu pendek (< 3 kata)."""
    assert is_valid_review("bad") is False
    assert is_valid_review("good game") is False
    assert is_valid_review("10/10") is False
    assert is_valid_review("Not recommended.") is False
    assert is_valid_review("") is False
    assert is_valid_review("   ") is False


def test_valid_normal_reviews():
    """Uji kelulusan ulasan gamer normal dengan campuran tanda baca dan konteks aspek."""
    review_1 = "Great graphics and smooth controls, but the technical bugs on boss fights make it frustrating."
    review_2 = "I loved the story and characters! Worth every penny at full price."
    review_3 = "Good game overall, but performance drops below 30 fps in the later stages."

    assert is_valid_review(review_1) is True
    assert is_valid_review(review_2) is True
    assert is_valid_review(review_3) is True
