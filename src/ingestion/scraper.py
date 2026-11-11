"""Steamworks API cursor-based review scraper with exponential backoff and resume support."""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SteamReviewScraper")


def _validate_max_reviews(max_reviews: Optional[int]) -> None:
    if max_reviews is not None and (
        isinstance(max_reviews, bool) or not isinstance(max_reviews, int) or max_reviews <= 0
    ):
        raise ValueError("max_reviews must be None or a positive integer")


def load_target_games_from_tsv(
    tsv_path: str | Path = "configs/appids_reference.tsv",
) -> List[Dict[str, Any]]:
    """Membaca berkas TSV referensi game dan mengembalikan list dict [{"app_id": int, "name": str}, ...].

    Args:
        tsv_path: Path menuju file TSV referensi (default configs/appids_reference.tsv).

    Returns:
        List berisi dictionary informasi game.
    """
    tsv_file = Path(tsv_path)
    if not tsv_file.exists():
        raise FileNotFoundError(f"Berkas referensi TSV tidak ditemukan: {tsv_file}")

    games: List[Dict[str, Any]] = []
    with open(tsv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            app_id_raw = row.get("appid") or row.get("app_id")
            if not app_id_raw or not app_id_raw.strip():
                continue
            name = row.get("judul") or row.get("name") or str(app_id_raw)
            games.append({"app_id": int(app_id_raw.strip()), "name": name.strip()})

    if not games:
        raise ValueError(f"Daftar target game kosong: {tsv_file}")

    logger.info("Berhasil memuat %d game dari %s", len(games), tsv_file)
    return games


class SteamReviewScraper:
    """Scraper untuk mengunduh ulasan game Steam via Steamworks API."""

    BASE_URL = "https://store.steampowered.com/appreviews/{app_id}"

    def __init__(
        self,
        config_path: Optional[str | Path] = "configs/config.yaml",
        filter: Optional[str] = None,
        language: Optional[str] = None,
        review_type: Optional[str] = None,
        purchase_type: Optional[str] = None,
        num_per_page: Optional[int] = None,
        delay_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        # 1. Base default values
        base_filter = "recent"
        base_language = "english"
        base_review_type = "all"
        base_purchase_type = "all"
        base_num_per_page = 100
        base_delay_seconds = 1.2
        base_max_retries = 5
        base_output_dir = "data/raw"

        # 2. Baca config.yaml terlebih dahulu jika ada
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

            scraper_cfg = cfg.get("scraper_params", {})
            base_filter = scraper_cfg.get("filter", base_filter)
            base_language = scraper_cfg.get("language", base_language)
            base_review_type = scraper_cfg.get("review_type", base_review_type)
            base_purchase_type = scraper_cfg.get("purchase_type", base_purchase_type)
            base_num_per_page = int(scraper_cfg.get("num_per_page", base_num_per_page))
            base_delay_seconds = float(scraper_cfg.get("delay_seconds", base_delay_seconds))
            base_max_retries = int(scraper_cfg.get("max_retries", base_max_retries))

            paths_cfg = cfg.get("paths", {})
            base_output_dir = paths_cfg.get("raw_dir", base_output_dir)

        # 3. Override dengan argumen eksplisit dari pemanggil jika disediakan
        self.params: Dict[str, Any] = {
            "json": 1,
            "filter": filter if filter is not None else base_filter,
            "language": language if language is not None else base_language,
            "review_type": review_type if review_type is not None else base_review_type,
            "purchase_type": purchase_type if purchase_type is not None else base_purchase_type,
            "num_per_page": num_per_page if num_per_page is not None else base_num_per_page,
        }
        self.delay_seconds = delay_seconds if delay_seconds is not None else base_delay_seconds
        self.max_retries = max_retries if max_retries is not None else base_max_retries
        self.output_dir = Path(output_dir if output_dir is not None else base_output_dir)

        self.backoff_delays = [2, 4, 8, 16, 32]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "SteamReviewResearch/1.0"
                )
            }
        )

    def fetch_page(
        self,
        app_id: int,
        cursor: str = "*",
    ) -> Optional[Dict[str, Any]]:
        """Mengambil satu halaman ulasan dengan retry dan exponential backoff.

        Args:
            app_id: Steam Application ID.
            cursor: Token kursor paginasi Steam.

        Returns:
            Dictionary JSON respons dari Steamworks API atau None jika gagal.
        """
        url = self.BASE_URL.format(app_id=app_id)
        request_params = dict(self.params)
        request_params["cursor"] = cursor

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=request_params, timeout=20)

                # HTTP 429: Too Many Requests
                if response.status_code == 429:
                    sleep_time = self.backoff_delays[
                        min(attempt, len(self.backoff_delays) - 1)
                    ]
                    logger.warning(
                        "HTTP 429 hit untuk app_id %d (attempt %d/%d). "
                        "Backoff %ds...",
                        app_id,
                        attempt + 1,
                        self.max_retries,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue

                response.raise_for_status()
                data = response.json()

                if (
                    not isinstance(data, dict)
                    or data.get("success") != 1
                    or not isinstance(data.get("reviews"), list)
                    or any(not isinstance(review, dict) for review in data["reviews"])
                ):
                    api_status = data.get("success") if isinstance(data, dict) else type(data).__name__
                    logger.warning(
                        "Steam API mengembalikan respons tidak valid atau non-sukses: %s",
                        api_status,
                    )
                    return None

                return data

            except (requests.RequestException, ValueError) as err:
                sleep_time = self.backoff_delays[
                    min(attempt, len(self.backoff_delays) - 1)
                ]
                logger.warning(
                    "Network error / parsing failure: %s (attempt %d/%d). "
                    "Backoff %ds...",
                    err,
                    attempt + 1,
                    self.max_retries,
                    sleep_time,
                )
                time.sleep(sleep_time)

        logger.error(
            "Gagal mengambil data app_id %d setelah %d retries.",
            app_id,
            self.max_retries,
        )
        return None

    def _extract_record(
        self,
        app_id: int,
        raw_review: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Mengekstrak dan memvalidasi skema baris ulasan mentah."""
        author = raw_review.get("author", {}) or {}
        return {
            "review_id": str(raw_review.get("recommendationid", "")),
            "app_id": int(app_id),
            "review_text": str(raw_review.get("review", "")),
            "voted_up": bool(raw_review.get("voted_up", False)),
            "timestamp_created": int(raw_review.get("timestamp_created") or 0),
            "author_playtime_forever": int(author.get("playtime_forever") or 0),
        }

    def scrape_game(
        self,
        app_id: int,
        max_reviews: Optional[int] = None,
        output_file: Optional[str | Path] = None,
        resume: bool = True,
    ) -> List[Dict[str, Any]]:
        """Mengunduh ulasan untuk 1 game dengan cursor pagination dan dukungan resume.

        Args:
            app_id: Steam Application ID.
            max_reviews: Batas jumlah ulasan yang diunduh (None untuk semua).
            output_file: Path spesifik berkas .jsonl (default data/raw/{app_id}.jsonl).
            resume: Jika True dan file sudah ada dengan jumlah ulasan >= max_reviews, lewati.

        Returns:
            List data ulasan yang tersimpan.
        """
        _validate_max_reviews(max_reviews)

        if output_file is None:
            output_file = self.output_dir / f"{app_id}.jsonl"
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        existing_records: List[Dict[str, Any]] = []
        valid_file_size = 0
        needs_newline = False
        if resume and output_file.exists():
            lines = output_file.read_bytes().splitlines(keepends=True)
            for index, line in enumerate(lines):
                if not line.strip():
                    valid_file_size += len(line)
                    needs_newline = not line.endswith((b"\n", b"\r"))
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"Invalid JSONL record in {output_file} at line {index + 1}")
                    existing_records.append(record)
                    valid_file_size += len(line)
                    needs_newline = not line.endswith((b"\n", b"\r"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if index != len(lines) - 1 or line.endswith((b"\n", b"\r")):
                        raise
                    logger.warning("Mengabaikan record JSONL terakhir yang terpotong pada %s", output_file)
                    break

            if max_reviews is not None and len(existing_records) >= max_reviews:
                if output_file.stat().st_size > valid_file_size:
                    with open(output_file, "r+b") as f_trunc:
                        f_trunc.truncate(valid_file_size)
                logger.info(
                    "Game %d sudah selesai discrape sebelumnya (%d >= target %d). Melewati (skip)...",
                    app_id,
                    len(existing_records),
                    max_reviews,
                )
                return existing_records[:max_reviews]

        cursor = "*"
        seen_cursors: set[str] = set()
        seen_ids = {record.get("review_id") for record in existing_records if record.get("review_id")}
        collected_reviews = existing_records
        page = 1

        logger.info(
            "Mulai scraping app_id: %d (target max: %s) -> %s",
            app_id,
            max_reviews if max_reviews is not None else "ALL",
            output_file,
        )

        if cursor != "*" and cursor in seen_cursors:
            logger.warning(
                "Terdeteksi siklus kursor berulang (%s) pada app_id %d. Menghentikan scraping game ini.",
                cursor,
                app_id,
            )
        seen_cursors.add(cursor)
        data = self.fetch_page(app_id=app_id, cursor=cursor)
        if data is None:
            raise RuntimeError(f"Failed to fetch reviews for app_id {app_id}")

        if resume and output_file.exists() and output_file.stat().st_size != valid_file_size:
            with open(output_file, "r+b") as f_out:
                f_out.truncate(valid_file_size)

        mode = "a" if resume and output_file.exists() else "w"
        with open(output_file, mode, encoding="utf-8") as f_out:
            if needs_newline and mode == "a":
                f_out.write("\n")
            while True:
                reviews = data.get("reviews", [])
                if not reviews:
                    logger.info("Tidak ada ulasan lagi yang ditemukan.")
                    break

                new_records_in_page = 0
                for raw_rev in reviews:
                    record = self._extract_record(app_id, raw_rev)
                    rev_id = record["review_id"]

                    if rev_id and rev_id in seen_ids:
                        continue
                    if rev_id:
                        seen_ids.add(rev_id)

                    collected_reviews.append(record)
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    new_records_in_page += 1

                    if max_reviews is not None and len(collected_reviews) >= max_reviews:
                        break

                logger.info(
                    "Page %d selesai: +%d ulasan baru (total: %d)",
                    page,
                    new_records_in_page,
                    len(collected_reviews),
                )

                if max_reviews is not None and len(collected_reviews) >= max_reviews:
                    logger.info("Target max_reviews (%d) tercapai.", max_reviews)
                    break

                next_cursor = data.get("cursor")
                if not next_cursor or next_cursor == cursor:
                    logger.info("Cursor selesai atau stagnan.")
                    break

                cursor = next_cursor
                page += 1
                time.sleep(self.delay_seconds)
                if cursor != "*" and cursor in seen_cursors:
                    logger.warning(
                        "Terdeteksi siklus kursor berulang (%s) pada app_id %d. Menghentikan scraping game ini.",
                        cursor,
                        app_id,
                    )
                    break
                seen_cursors.add(cursor)
                data = self.fetch_page(app_id=app_id, cursor=cursor)
                if data is None:
                    raise RuntimeError(f"Failed to fetch reviews for app_id {app_id}")

        logger.info(
            "Scraping selesai untuk app_id %d. Tersimpan %d ulasan di %s",
            app_id,
            len(collected_reviews),
            output_file,
        )
        return collected_reviews

    def scrape_all(
        self,
        target_games: Optional[List[Dict[str, Any]]] = None,
        max_reviews_per_game: int = 500,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Menjalankan scraping batch untuk list game dengan progress bar tqdm.

        Toleran terhadap network drop: game yang gagal dicatat ke log error
        tanpa menghentikan iterasi game lainnya.

        Args:
            target_games: List dict game [{"app_id": int, "name": str}, ...].
            max_reviews_per_game: Batas ulasan per game (default 500).
            resume: Lewati game yang sudah lengkap diunduh.

        Returns:
            Dict ringkasan hasil batch: {"successful": list, "failed": list, "total": int}.
        """
        _validate_max_reviews(max_reviews_per_game)
        if target_games is None:
            tsv_path = Path("configs/appids_reference.tsv")
            if tsv_path.exists():
                target_games = load_target_games_from_tsv(tsv_path)
            else:
                logger.warning("TSV referensi tidak ditemukan, menggunakan target_games dari config.")
                with open("configs/config.yaml", "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                target_games = cfg.get("target_games", [])

        if not target_games:
            raise ValueError("target_games must not be empty")

        successful_games: List[Dict[str, Any]] = []
        failed_games: List[Dict[str, Any]] = []

        logger.info(
            "Memulai batch scraping untuk %d game (target per game: %s)",
            len(target_games),
            max_reviews_per_game if max_reviews_per_game is not None else "ALL",
        )

        with tqdm(target_games, desc="Batch Scraping Games", unit="game") as pbar:
            for game in pbar:
                app_id = int(game["app_id"])
                game_name = game.get("name", str(app_id))
                pbar.set_postfix_str(f"{game_name[:20]}")

                try:
                    reviews = self.scrape_game(
                        app_id=app_id,
                        max_reviews=max_reviews_per_game,
                        resume=resume,
                    )
                    successful_games.append(
                        {"app_id": app_id, "name": game_name, "count": len(reviews)}
                    )
                except Exception as exc:
                    logger.error(
                        "Gagal melakukan scraping pada game %s (app_id %d): %s",
                        game_name,
                        app_id,
                        exc,
                    )
                    failed_games.append(
                        {"app_id": app_id, "name": game_name, "error": str(exc)}
                    )

        logger.info(
            "Batch scraping selesai. Sukses: %d, Gagal: %d, Total: %d",
            len(successful_games),
            len(failed_games),
            len(target_games),
        )
        return {
            "successful": successful_games,
            "failed": failed_games,
            "total": len(target_games),
        }
