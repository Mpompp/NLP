"""Game Scorecard and Aspect Satisfaction Index (ASI) Roll-Up Generator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ScorecardEngine")


def determine_diagnostic_flag(
    gameplay_asi: Optional[float],
    story_asi: Optional[float],
    technical_asi: Optional[float],
) -> str:
    """Menentukan flag diagnostik otomatis berdasarkan nilai ASI ketiga aspek."""
    if gameplay_asi is None or story_asi is None or technical_asi is None:
        return "Data Kurang"
    if np.isnan(gameplay_asi) or np.isnan(story_asi) or np.isnan(technical_asi):
        return "Data Kurang"

    if technical_asi < 40.0:
        return "ALERT: Stabilitas Teknis Kritis (Crash/Bug)"
    if gameplay_asi >= 85.0 and story_asi >= 85.0:
        return "Pujian Kreatif & Mekanik Sangat Tinggi"
    if gameplay_asi < 50.0:
        return "Keluhan Mekanik/Kontrol Dominan"
    return "Karakteristik Seimbang"


def generate_game_scorecard(
    predictions_path: Union[str, Path] = "data/clean/corpus_predictions.csv",
    appids_ref_path: Union[str, Path] = "configs/appids_reference.tsv",
    output_path: Union[str, Path] = "reports/game_scorecards.csv",
    min_reviews_threshold: int = 10,
) -> pd.DataFrame:
    """Mengagregasi prediksi ulasan menjadi Game Scorecard (Aspect Satisfaction Index / ASI) per gim.

    Formula ASI:
        ASI_a = (Positif / (Positif + Negatif)) * 100%

    Args:
        predictions_path: Path berkas CSV corpus_predictions.
        appids_ref_path: Path berkas TSV referensi metadata gim (appid, judul).
        output_path: Path berkas CSV output scorecard.
        min_reviews_threshold: Ambang batas minimal ulasan polar (pos+neg) agar ASI valid.

    Returns:
        pd.DataFrame Game Scorecard.
    """
    pred_file = Path(predictions_path)
    ref_file = Path(appids_ref_path)
    out_file = Path(output_path)

    if not pred_file.exists():
        raise FileNotFoundError(f"Berkas prediksi ulasan tidak ditemukan: {pred_file}")

    # 1. Muat metadata referensi judul gim jika ada
    name_map: Dict[int, str] = {}
    if ref_file.exists():
        try:
            ref_df = pd.read_csv(ref_file, sep="\t", encoding="utf-8-sig")
            if "appid" in ref_df.columns and "judul" in ref_df.columns:
                for _, row in ref_df.iterrows():
                    try:
                        raw_title = str(row["judul"]).strip()
                        if not raw_title or raw_title.lower() == "nan":
                            name_map[int(row["appid"])] = f"App ID {int(row['appid'])}"
                        else:
                            name_map[int(row["appid"])] = raw_title
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            logger.warning("Gagal membaca referensi TSV %s: %s", ref_file, e)

    # 2. Muat hasil prediksi
    df = pd.read_csv(pred_file)
    if "app_id" not in df.columns or df["app_id"].isnull().any():
        raise ValueError("Ditemukan baris prediksi dengan app_id bernilai null/NaN. Skor scorecard tidak dapat diagregasi.")

    if len(df) == 0:
        logger.warning("Data prediksi kosong!")
        empty_df = pd.DataFrame(
            columns=[
                "app_id",
                "game_name",
                "total_reviews",
                "gameplay_asi",
                "story_asi",
                "technical_asi",
                "diagnostic_flag",
            ]
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(out_file, index=False, encoding="utf-8-sig")
        return empty_df

    # 3. Kelompokkan per gim dan hitung ASI per aspek
    records: List[Dict[str, Any]] = []
    grouped = df.groupby("app_id")

    for app_id, group in grouped:
        app_id_int = int(app_id)
        game_name = name_map.get(app_id_int, f"App ID {app_id_int}")
        total_reviews = len(group)

        asi_values: Dict[str, Optional[float]] = {}
        for aspect in ("gameplay", "story", "technical"):
            col = f"pred_{aspect}"
            if col not in group.columns:
                asi_values[aspect] = None
                continue

            # Kelas 1: pos, Kelas 2: neg, Kelas 0: none (diabaikan)
            pos_count = int((group[col] == 1).sum())
            neg_count = int((group[col] == 2).sum())
            relevant_count = pos_count + neg_count

            if relevant_count == 0 or relevant_count < min_reviews_threshold:
                asi_values[aspect] = None
            else:
                asi = (pos_count / relevant_count) * 100.0
                asi_values[aspect] = round(float(asi), 2)

        flag = determine_diagnostic_flag(
            gameplay_asi=asi_values["gameplay"],
            story_asi=asi_values["story"],
            technical_asi=asi_values["technical"],
        )

        records.append(
            {
                "app_id": app_id_int,
                "game_name": game_name,
                "total_reviews": total_reviews,
                "gameplay_asi": asi_values["gameplay"],
                "story_asi": asi_values["story"],
                "technical_asi": asi_values["technical"],
                "diagnostic_flag": flag,
            }
        )

    scorecard_df = pd.DataFrame(records)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    scorecard_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    logger.info("Game Scorecard berhasil digenerate ke %s (%d gim)", out_file, len(scorecard_df))
    return scorecard_df


def format_scorecard_summary(scorecard_df: pd.DataFrame, top_n: int = 5) -> str:
    """Format ringkasan scorecard menjadi tabel konsol ASCII informatif."""
    if len(scorecard_df) == 0:
        return "Scorecard kosong."

    header = (
        "+---------+--------------------------------+---------+----------+----------+----------+-----------------------------------------------+\n"
        "| App ID  | Judul Gim                      | Ulasan  | ASI_Game | ASI_Stor | ASI_Tech | Diagnostic Flag                       |\n"
        "+---------+--------------------------------+---------+----------+----------+----------+-----------------------------------------------+"
    )
    separator = (
        "+---------+--------------------------------+---------+----------+----------+----------+-----------------------------------------------+"
    )
    rows = [header]

    # Ambil sampel gim: top_n teratas + semua gim yang mendapat alert kritis
    alerts = scorecard_df[scorecard_df["diagnostic_flag"].astype(str).str.startswith("ALERT")]
    sample_df = pd.concat([scorecard_df.head(top_n), alerts]).drop_duplicates(subset=["app_id"])

    for _, row in sample_df.iterrows():
        app_id = str(row["app_id"])
        name = str(row["game_name"])[:30]
        total = str(row["total_reviews"])
        gp = f"{row['gameplay_asi']:.1f}%" if pd.notnull(row["gameplay_asi"]) else "N/A"
        st = f"{row['story_asi']:.1f}%" if pd.notnull(row["story_asi"]) else "N/A"
        tc = f"{row['technical_asi']:.1f}%" if pd.notnull(row["technical_asi"]) else "N/A"
        flag = str(row["diagnostic_flag"])[:45]

        row_str = (
            f"| {app_id:<7} | {name:<30} | {total:>7} | {gp:>8} | {st:>8} | {tc:>8} | {flag:<45} |"
        )
        rows.append(row_str)

    rows.append(separator)
    return "\n".join(rows)
