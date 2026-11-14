"""Master CLI entry point for Steam Aspect-Category Sentiment Analysis (ACSA) pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ACSA-CLI")
CONFIG_PATH = Path("configs/config.yaml")


def get_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def get_ground_truth_path(
    custom_path: Optional[str] = None, mock: bool = False
) -> Path:
    """Return the explicitly selected ground-truth file without mock fallback."""
    if custom_path:
        path = Path(custom_path)
        if not path.is_file():
            raise FileNotFoundError(f"Berkas ground truth tidak ditemukan: {path}")
        return path
    annotation_dir = Path(get_config().get("paths", {}).get("annotation_dir", "data/annotation"))
    path = annotation_dir / ("ground_truth_test.csv" if mock else "ground_truth.csv")
    if not path.is_file() and mock:
        from src.dataset.mock import generate_mock_ground_truth

        generate_mock_ground_truth(output_path=path)
    if not path.is_file():
        if mock:
            raise FileNotFoundError(f"Berkas ground truth dummy tidak ditemukan setelah dibuat: {path}")
        raise FileNotFoundError(
            f"File ground truth riil belum ditemukan di {path}. "
            "Silakan lakukan anotasi terlebih dahulu atau gunakan flag --mock jika sengaja ingin menguji data dummy."
        )
    return path


def get_paths() -> dict:
    return get_config().get("paths", {})


def configured_path(key: str, default: str) -> Path:
    return Path(get_paths().get(key, default))


def add_ground_truth_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ground-truth", type=str, default=None, help="Path khusus CSV ground truth")
    parser.add_argument("--mock", action="store_true", help="Gunakan ground truth dummy untuk pengujian")


def require_ground_truth(args: argparse.Namespace) -> Path:
    return get_ground_truth_path(args.ground_truth, args.mock)


def require_split_path() -> Path:
    split_path = configured_path("splits_dir", "data/splits") / "split_indices.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Berkas {split_path} tidak ditemukan. Jalankan 'python run.py split' terlebih dahulu.")
    return split_path


def cmd_scrape(args: argparse.Namespace) -> None:
    """Menjalankan scraper untuk 150 gim target."""
    from src.ingestion.scraper import SteamReviewScraper, load_target_games_from_tsv

    tsv_path = Path(get_config().get("target_150_reference", "configs/appids_reference.tsv"))
    if not tsv_path.exists():
        logger.error("Berkas referensi 150 gim %s tidak ditemukan!", tsv_path)
        sys.exit(1)

    if args.max_reviews is not None and args.max_reviews <= 0:
        raise ValueError("max_reviews must be None or a positive integer")

    games = load_target_games_from_tsv(tsv_path)
    logger.info("Memulai scraping untuk %d gim (max %s ulasan/gim)...", len(games), args.max_reviews)

    scraper = SteamReviewScraper()
    scraper.scrape_all(target_games=games, max_reviews_per_game=args.max_reviews, resume=True)


def cmd_clean(args: argparse.Namespace) -> None:
    """Membangun korpus bersih dari seluruh berkas JSONL mentah."""
    from src.dataset.sampler import build_clean_corpus

    raw_dir = configured_path("raw_dir", "data/raw")
    clean_path = configured_path("clean_dir", "data/clean") / "corpus_clean.csv"

    if not raw_dir.exists() or len(list(raw_dir.glob("*.jsonl"))) == 0:
        logger.error("Direktori %s kosong atau tidak ditemukan. Jalankan 'scrape' terlebih dahulu!", raw_dir)
        sys.exit(1)

    build_clean_corpus(raw_dir=raw_dir, output_path=clean_path)


def cmd_sample(args: argparse.Namespace) -> None:
    """Membuat sampel ulasan untuk anotasi manual ground truth."""
    from src.dataset.sampler import generate_annotation_sample

    corpus_path = configured_path("clean_dir", "data/clean") / "corpus_clean.csv"
    out_path = configured_path("annotation_dir", "data/annotation") / "sample_anotasi.csv"

    if not corpus_path.exists():
        logger.error("Korpus bersih %s tidak ditemukan. Jalankan 'clean' terlebih dahulu!", corpus_path)
        sys.exit(1)

    generate_annotation_sample(corpus_path=corpus_path, output_path=out_path, n_samples=args.n_samples, seed=42)


def cmd_split(args: argparse.Namespace) -> None:
    """Membagi ground truth berlabel menjadi partisi anti-leakage 70/15/15."""
    from src.dataset.split import create_stratified_split

    gt_path = require_ground_truth(args)
    split_path = configured_path("splits_dir", "data/splits") / "split_indices.json"
    create_stratified_split(ground_truth_path=gt_path, output_split_path=split_path, seed=42)


def cmd_train(args: argparse.Namespace) -> None:
    """Melatih model BERT per aspek atau seluruh aspek secara sekuensial."""
    from src.training.trainer import train_all_aspects, train_aspect_model

    gt_path = require_ground_truth(args)
    split_path = require_split_path()

    if args.aspect == "all":
        train_all_aspects(
            ground_truth_path=gt_path,
            split_indices_path=split_path,
            models_dir=configured_path("models_dir", "models"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    else:
        train_aspect_model(
            aspect_name=args.aspect,
            ground_truth_path=gt_path,
            split_indices_path=split_path,
            models_dir=configured_path("models_dir", "models"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )


def cmd_baseline(args: argparse.Namespace) -> None:
    """Melatih pembanding baseline Scikit-Learn Linear SVM."""
    from src.models.baseline_svm import train_all_baselines

    gt_path = require_ground_truth(args)
    split_path = require_split_path()
    train_all_baselines(
        ground_truth_path=gt_path,
        split_indices_path=split_path,
        models_dir=configured_path("models_dir", "models"),
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Menjalankan evaluasi komparatif apple-to-apple antara SVM vs BERT pada Test Set identik."""
    from src.training.evaluate import evaluate_test_set_comparison

    gt_path = require_ground_truth(args)
    split_path = require_split_path()
    aspects = None if args.aspect == "all" else [args.aspect]
    evaluate_test_set_comparison(
        ground_truth_path=gt_path,
        split_indices_path=split_path,
        models_dir=configured_path("models_dir", "models"),
        reports_dir=configured_path("reports_dir", "reports"),
        aspects=aspects,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    """Menjalankan inferensi massal 3 model BERT pada ulasan unseen."""
    from src.inference.predict import predict_unseen_corpus

    clean_dir = configured_path("clean_dir", "data/clean")
    corpus_path = clean_dir / "corpus_clean.csv"
    split_path = require_split_path()
    out_path = clean_dir / "corpus_predictions.csv"

    if not corpus_path.exists():
        logger.error("Korpus bersih %s tidak ditemukan!", corpus_path)
        sys.exit(1)

    predict_unseen_corpus(
        corpus_path=corpus_path,
        split_indices_path=split_path,
        models_dir=configured_path("models_dir", "models"),
        output_path=out_path,
        batch_size=args.batch_size,
    )


def cmd_scorecard(args: argparse.Namespace) -> None:
    """Mengagregasi prediksi ulasan menjadi Game Scorecard (ASI) dan mencetak ringkasan."""
    from src.inference.scorecard import format_scorecard_summary, generate_game_scorecard

    pred_path = configured_path("clean_dir", "data/clean") / "corpus_predictions.csv"
    ref_path = Path(get_config().get("target_150_reference", "configs/appids_reference.tsv"))
    out_path = configured_path("reports_dir", "reports") / "game_scorecards.csv"

    if not pred_path.exists():
        logger.error("Berkas prediksi %s tidak ditemukan! Jalankan 'predict' terlebih dahulu.", pred_path)
        sys.exit(1)

    scorecard_df = generate_game_scorecard(
        predictions_path=pred_path,
        appids_ref_path=ref_path,
        output_path=out_path,
        min_reviews_threshold=args.min_reviews,
    )

    summary_table = format_scorecard_summary(scorecard_df, top_n=args.top_n)
    print("\n" + summary_table + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Membangun parser CLI utama beserta seluruh sub-command-nya."""
    parser = argparse.ArgumentParser(
        description="Steam Aspect-Category Sentiment Analysis (ACSA) Master CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Perintah tahapan pipeline")

    # 1. Scrape
    p_scrape = subparsers.add_parser("scrape", help="Scraping ulasan ulasan Steam untuk 150 gim indie target")
    p_scrape.add_argument("--max-reviews", type=int, default=500, help="Maksimal ulasan per game")
    p_scrape.set_defaults(func=cmd_scrape)

    # 2. Clean
    p_clean = subparsers.add_parser("clean", help="Membersihkan tag BBCode dan menyaring ASCII art dari data raw")
    p_clean.set_defaults(func=cmd_clean)

    # 3. Sample
    p_sample = subparsers.add_parser("sample", help="Menghasilkan sampel terstratifikasi untuk anotasi manual")
    p_sample.add_argument("--n-samples", type=int, default=1500, help="Jumlah total sampel ulasan")
    p_sample.set_defaults(func=cmd_sample)

    # 4. Split
    p_split = subparsers.add_parser("split", help="Partisi anti-leakage 70/15/15 pada ground truth berlabel")
    add_ground_truth_arguments(p_split)
    p_split.set_defaults(func=cmd_split)

    # 5. Train
    p_train = subparsers.add_parser("train", help="Melatih model BERT per aspek dengan manual PyTorch loop")
    p_train.add_argument("--aspect", choices=["all", "gameplay", "story", "technical"], default="all")
    p_train.add_argument("--epochs", type=int, default=3, help="Jumlah epoch pelatihan")
    p_train.add_argument("--batch-size", type=int, default=16, help="Batch size DataLoader")
    p_train.add_argument("--lr", type=float, default=2e-5, help="Learning rate AdamW")
    add_ground_truth_arguments(p_train)
    p_train.set_defaults(func=cmd_train)

    # 6. Baseline
    p_base = subparsers.add_parser("baseline", help="Melatih pembanding model Scikit-Learn Linear SVM")
    add_ground_truth_arguments(p_base)
    p_base.set_defaults(func=cmd_baseline)

    # 7. Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluasi komparatif SVM vs BERT pada Test Set identik")
    add_ground_truth_arguments(p_eval)
    p_eval.add_argument("--aspect", choices=["all", "gameplay", "story", "technical"], default="all")
    p_eval.set_defaults(func=cmd_evaluate)

    # 8. Predict
    p_pred = subparsers.add_parser("predict", help="Inferensi massal 3 model BERT pada korpus ulasan unseen")
    p_pred.add_argument("--batch-size", type=int, default=32, help="Batch size inferensi")
    p_pred.set_defaults(func=cmd_predict)

    # 9. Scorecard
    p_score = subparsers.add_parser("scorecard", help="Agregasi skor kepuasan gim (ASI) dan laporan diagnostik")
    p_score.add_argument("--min-reviews", type=int, default=10, help="Batas minimum ulasan relevan per aspek")
    p_score.add_argument("--top-n", type=int, default=10, help="Jumlah sampel gim di tabel konsol")
    p_score.set_defaults(func=cmd_scorecard)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
