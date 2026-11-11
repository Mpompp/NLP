"""Deterministic smoke tests for SteamReviewScraper."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from src.ingestion.scraper import SteamReviewScraper, load_target_games_from_tsv


def _response(data, status_code=200):
    return SimpleNamespace(
        status_code=status_code,
        raise_for_status=lambda: None,
        json=lambda: data,
    )


def _review(review_id, **values):
    return {
        "recommendationid": review_id,
        "review": f"Review {review_id}",
        "voted_up": True,
        "timestamp_created": 123,
        "author": {"playtime_forever": 10},
        **values,
    }


def test_scraper_smoke_mocked_pagination_and_resume(tmp_path, monkeypatch):
    app_id = 504230
    output_file = tmp_path / f"{app_id}.jsonl"
    calls = []
    pages = {
        "*": {"success": 1, "cursor": "next", "reviews": [_review("1"), _review("2")]},
        "next": {
            "success": 1,
            "cursor": "last",
            "reviews": [
                _review("3", timestamp_created=None, author={"playtime_forever": None}),
                _review("4"),
            ],
        },
    }

    def mock_get(self, url, *, params, timeout):
        calls.append(params["cursor"])
        return _response(pages[params["cursor"]])

    monkeypatch.setattr(requests.Session, "get", mock_get)
    monkeypatch.setattr("src.ingestion.scraper.time.sleep", lambda _: None)
    scraper = SteamReviewScraper(
        config_path=None,
        num_per_page=2,
        delay_seconds=0,
        output_dir=tmp_path,
    )

    reviews = scraper.scrape_game(app_id, max_reviews=3, output_file=output_file)
    assert len(reviews) == 3
    assert calls == ["*", "next"]
    assert reviews[2]["timestamp_created"] == 0
    assert reviews[2]["author_playtime_forever"] == 0
    assert len(output_file.read_text(encoding="utf-8").splitlines()) == 3

    resumed = scraper.scrape_game(app_id, max_reviews=3, output_file=output_file)
    assert len(resumed) == 3
    assert calls == ["*", "next"]


@pytest.mark.parametrize(
    "response_data",
    [
        {"success": 2, "reviews": []},
        {"success": 1},
        {"success": 1, "reviews": [None]},
    ],
)
def test_fetch_page_rejects_unsuccessful_or_malformed_api_response(
    tmp_path, monkeypatch, response_data
):
    monkeypatch.setattr(
        requests.Session,
        "get",
        lambda *_args, **_kwargs: _response(response_data),
    )
    scraper = SteamReviewScraper(config_path=None, output_dir=tmp_path, max_retries=1)

    assert scraper.fetch_page(504230) is None


def test_fetch_page_retries_with_backoff(tmp_path, monkeypatch):
    calls = []
    sleeps = []

    def mock_get(self, url, *, params, timeout):
        calls.append(params["cursor"])
        if len(calls) == 1:
            raise requests.ConnectionError("temporary failure")
        if len(calls) == 2:
            return _response({}, status_code=429)
        return _response({"success": 1, "reviews": []})

    monkeypatch.setattr(requests.Session, "get", mock_get)
    monkeypatch.setattr("src.ingestion.scraper.time.sleep", sleeps.append)
    scraper = SteamReviewScraper(config_path=None, output_dir=tmp_path, max_retries=3)

    assert scraper.fetch_page(504230) == {"success": 1, "reviews": []}
    assert calls == ["*", "*", "*"]
    assert sleeps == [2, 4]


def test_scrape_all_marks_request_failure_failed_and_preserves_partial_jsonl(
    tmp_path, monkeypatch
):
    app_id = 504230
    output_file = tmp_path / f"{app_id}.jsonl"
    original = {
        "review_id": "existing",
        "app_id": app_id,
        "review_text": "Saved before resume",
        "voted_up": True,
        "timestamp_created": 1,
        "author_playtime_forever": 0,
    }
    output_file.write_text(json.dumps(original) + "\n", encoding="utf-8")
    calls = []

    def mock_get(self, url, *, params, timeout):
        calls.append(params["cursor"])
        if params["cursor"] == "*":
            return _response(
                {
                    "success": 1,
                    "cursor": "next",
                    "reviews": [_review("existing"), _review("new")],
                }
            )
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests.Session, "get", mock_get)
    monkeypatch.setattr("src.ingestion.scraper.time.sleep", lambda _: None)
    scraper = SteamReviewScraper(
        config_path=None,
        output_dir=tmp_path,
        max_retries=1,
        delay_seconds=0,
    )

    result = scraper.scrape_all([{"app_id": app_id, "name": "Game"}], max_reviews_per_game=3)

    assert result["successful"] == []
    assert result["failed"][0]["app_id"] == app_id
    assert calls == ["*", "next"]
    records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert [record["review_id"] for record in records] == ["existing", "new"]


def test_scrape_game_continues_after_duplicate_only_page(tmp_path, monkeypatch):
    cursors = []
    pages = {
        "*": {"success": 1, "cursor": "next", "reviews": [_review("same")]},
        "next": {"success": 1, "cursor": "last", "reviews": [_review("same")]},
        "last": {"success": 1, "reviews": [_review("new")]},
    }

    def mock_get(self, url, *, params, timeout):
        cursors.append(params["cursor"])
        return _response(pages[params["cursor"]])

    monkeypatch.setattr(requests.Session, "get", mock_get)
    monkeypatch.setattr("src.ingestion.scraper.time.sleep", lambda _: None)
    output_file = tmp_path / "504230.jsonl"
    output_file.write_text("stale data\n", encoding="utf-8")
    scraper = SteamReviewScraper(config_path=None, output_dir=tmp_path, delay_seconds=0)

    reviews = scraper.scrape_game(504230, max_reviews=2, output_file=output_file, resume=False)

    assert cursors == ["*", "next", "last"]
    assert [review["review_id"] for review in reviews] == ["same", "new"]


def test_scrape_game_recovers_from_truncated_final_jsonl_record(tmp_path, monkeypatch):
    output_file = tmp_path / "504230.jsonl"
    original = {
        "review_id": "existing",
        "app_id": 504230,
        "review_text": "Review existing",
        "voted_up": True,
        "timestamp_created": 123,
        "author_playtime_forever": 10,
    }
    output_file.write_text(json.dumps(original) + '\n{"review_id":"broken', encoding="utf-8")
    scraper = SteamReviewScraper(config_path=None, output_dir=tmp_path, delay_seconds=0)
    monkeypatch.setattr(
        scraper,
        "fetch_page",
        lambda **_: {"success": 1, "cursor": "next", "reviews": [_review("new")]},
    )

    records = scraper.scrape_game(504230, max_reviews=2, output_file=output_file)

    assert [record["review_id"] for record in records] == ["existing", "new"]
    assert [json.loads(line)["review_id"] for line in output_file.read_text(encoding="utf-8").splitlines()] == [
        "existing",
        "new",
    ]


def test_scrape_game_preserves_existing_file_when_initial_fetch_fails(tmp_path, monkeypatch):
    output_file = tmp_path / "504230.jsonl"
    original = "existing data\n"
    output_file.write_text(original, encoding="utf-8")
    scraper = SteamReviewScraper(config_path=None, output_dir=tmp_path)
    monkeypatch.setattr(scraper, "fetch_page", lambda **_: None)

    with pytest.raises(RuntimeError, match="Failed to fetch"):
        scraper.scrape_game(504230, output_file=output_file, resume=False)

    assert output_file.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("max_reviews", [0, -1, True, 1.5, "2"])
def test_scrape_game_rejects_invalid_max_reviews(tmp_path, max_reviews):
    scraper = SteamReviewScraper(config_path=None, output_dir=tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        scraper.scrape_game(504230, max_reviews=max_reviews)


def test_scrape_all_rejects_empty_target_list(tmp_path):
    scraper = SteamReviewScraper(config_path=None, output_dir=tmp_path)
    with pytest.raises(ValueError, match="must not be empty"):
        scraper.scrape_all(target_games=[])


def test_scrape_all_rejects_invalid_review_limit_before_processing_games(tmp_path, monkeypatch):
    scraper = SteamReviewScraper(config_path=None, output_dir=tmp_path)
    monkeypatch.setattr(scraper, "scrape_game", lambda **_: pytest.fail("scrape_game must not run"))

    with pytest.raises(ValueError, match="positive integer"):
        scraper.scrape_all(
            target_games=[{"app_id": 504230, "name": "Celeste"}],
            max_reviews_per_game=0,
        )


def test_load_target_games_from_utf8_bom_tsv(tmp_path):
    tsv_path = tmp_path / "games.tsv"
    tsv_path.write_text("﻿appid\tjudul\n504230\tCeleste\n", encoding="utf-8")
    assert load_target_games_from_tsv(tsv_path) == [{"app_id": 504230, "name": "Celeste"}]


def test_load_target_games_rejects_empty_tsv(tmp_path):
    tsv_path = tmp_path / "games.tsv"
    tsv_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Daftar target game kosong"):
        load_target_games_from_tsv(tsv_path)


def test_load_target_150_games_from_tsv():
    """Verify the project reference TSV still contains 150 indie games."""
    tsv_path = Path("configs/appids_reference.tsv")
    assert tsv_path.exists(), f"File referensi TSV {tsv_path} tidak ditemukan"

    games = load_target_games_from_tsv(tsv_path)
    assert len(games) == 150
    assert all(isinstance(game["app_id"], int) for game in games)
    assert all(isinstance(game["name"], str) and game["name"] for game in games)


def test_scrape_game_stops_on_cyclic_cursor_loop(tmp_path, monkeypatch):
    """Verifikasi bahwa scraper berhenti jika mendeteksi siklus kursor berulang (A -> B -> A)."""
    cursors = []
    pages = {
        "*": {"success": 1, "cursor": "cur_a", "reviews": [_review("r1")]},
        "cur_a": {"success": 1, "cursor": "cur_b", "reviews": [_review("r2")]},
        "cur_b": {"success": 1, "cursor": "cur_a", "reviews": [_review("r3")]},
    }

    def mock_get(self, url, *, params, timeout):
        cursors.append(params["cursor"])
        return _response(pages[params["cursor"]])

    monkeypatch.setattr(requests.Session, "get", mock_get)
    monkeypatch.setattr("src.ingestion.scraper.time.sleep", lambda _: None)
    output_file = tmp_path / "504230.jsonl"
    scraper = SteamReviewScraper(config_path=None, output_dir=tmp_path, delay_seconds=0)

    reviews = scraper.scrape_game(504230, max_reviews=100, output_file=output_file, resume=False)

    # Harus berhenti setelah cur_a terdeteksi kembali: * -> cur_a -> cur_b -> (cur_a cycle detected -> break)
    assert cursors == ["*", "cur_a", "cur_b"]
    assert [review["review_id"] for review in reviews] == ["r1", "r2", "r3"]


def test_scrape_game_truncates_corrupt_tail_when_target_already_reached(tmp_path):
    """Verifikasi bahwa record rusak di akhir file tetap dipotong saat resume mencapai max_reviews."""
    output_file = tmp_path / "504230.jsonl"
    rec1 = {"review_id": "r1", "app_id": 504230, "review_text": "text1", "voted_up": True, "timestamp_created": 1, "author_playtime_forever": 1}
    rec2 = {"review_id": "r2", "app_id": 504230, "review_text": "text2", "voted_up": True, "timestamp_created": 2, "author_playtime_forever": 2}
    valid_content = json.dumps(rec1) + "\n" + json.dumps(rec2) + "\n"
    corrupt_content = valid_content + '{"review_id":"corrupt_without_newline'
    output_file.write_text(corrupt_content, encoding="utf-8")

    scraper = SteamReviewScraper(config_path=None, output_dir=tmp_path, delay_seconds=0)
    reviews = scraper.scrape_game(504230, max_reviews=2, output_file=output_file, resume=True)

    assert len(reviews) == 2
    # Pastikan file telah ditruncate sehingga teks korup hilang
    assert output_file.read_text(encoding="utf-8") == valid_content

