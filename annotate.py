"""CLI Interactive Annotation Tool for Steam Review ACSA.

Mempermudah anotasi manual 1.500 ulasan dengan input 3 angka cepat (gameplay, story, tech).
Mendukung auto-resume, auto-save setiap baris, dan fitur undo.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

SAMPLE_PATH = Path("data/annotation/sample_anotasi.csv")
GROUND_TRUTH_PATH = Path("data/annotation/ground_truth.csv")
TSV_PATH = Path("configs/appids_reference.tsv")

VALID_LABELS = {"0", "1", "2"}


def load_game_names() -> Dict[int, str]:
    """Memuat mapping app_id -> judul gim dari TSV referensi."""
    name_map = {}
    if TSV_PATH.exists():
        try:
            with open(TSV_PATH, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    appid_str = row.get("appid")
                    title = row.get("judul") or row.get("name")
                    if appid_str and title:
                        name_map[int(appid_str.strip())] = title.strip()
        except Exception:
            pass
    return name_map


def clear_screen() -> None:
    """Membersihkan layar terminal."""
    os.system("cls" if os.name == "nt" else "clear")


def main() -> None:
    if not SAMPLE_PATH.exists():
        print(f"Error: Berkas sampel {SAMPLE_PATH} tidak ditemukan!")
        print("Jalankan 'python run.py sample --n-samples 1500' terlebih dahulu.")
        sys.exit(1)

    name_map = load_game_names()

    # Baca sample awal
    sample_df = pd.read_csv(SAMPLE_PATH, dtype={"review_id": str})

    # Jika ground_truth.csv sudah ada sebelumnya, resume progres anotasi
    if GROUND_TRUTH_PATH.exists():
        gt_df = pd.read_csv(GROUND_TRUTH_PATH, dtype={"review_id": str})
        # Gabungkan progres yang sudah ada
        labeled_map = {}
        for _, row in gt_df.iterrows():
            g = str(row.get("gameplay", "")).strip()
            s = str(row.get("story", "")).strip()
            t = str(row.get("technical", "")).strip()
            if g in VALID_LABELS and s in VALID_LABELS and t in VALID_LABELS:
                labeled_map[str(row["review_id"])] = (int(g), int(s), int(t))

        for idx, row in sample_df.iterrows():
            rev_id = str(row["review_id"])
            if rev_id in labeled_map:
                g, s, t = labeled_map[rev_id]
                sample_df.at[idx, "gameplay"] = g
                sample_df.at[idx, "story"] = s
                sample_df.at[idx, "technical"] = t
    else:
        sample_df["gameplay"] = ""
        sample_df["story"] = ""
        sample_df["technical"] = ""

    total_rows = len(sample_df)

    def is_annotated(idx: int) -> bool:
        g = str(sample_df.at[idx, "gameplay"]).strip()
        s = str(sample_df.at[idx, "story"]).strip()
        t = str(sample_df.at[idx, "technical"]).strip()
        return g in VALID_LABELS and s in VALID_LABELS and t in VALID_LABELS

    def save_progress() -> None:
        GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(GROUND_TRUTH_PATH, index=False, encoding="utf-8-sig")

    # Temukan baris pertama yang belum teranotasi
    current_idx = 0
    while current_idx < total_rows and is_annotated(current_idx):
        current_idx += 1

    history: List[int] = []

    while current_idx < total_rows:
        completed_count = sum(1 for i in range(total_rows) if is_annotated(i))
        pct = (completed_count / total_rows) * 100.0

        row = sample_df.iloc[current_idx]
        app_id = int(row.get("app_id", 0))
        game_title = name_map.get(app_id, f"App ID {app_id}")
        review_text = str(row.get("clean_text", "")).strip()

        clear_screen()
        print("=" * 75)
        print(f" STEAM ACSA ANNOTATION TOOL  |  Progres: {completed_count}/{total_rows} ({pct:.1f}%)")
        print("=" * 75)
        print(f"Gim      : {game_title} (ID: {app_id})")
        print(f"Ulasan ID: {row.get('review_id')}")
        print("-" * 75)
        print(review_text)
        print("-" * 75)
        print("Panduan Nilai: 0 = None (tidak dibahas) | 1 = Positif | 2 = Negatif")
        print("Urutan Aspek : [1] Gameplay   [2] Story & UX   [3] Technical")
        print("Perintah Lain: 'u' = Undo baris sebelumnya | 's' = Skip | 'q' = Simpan & Keluar")
        print("-" * 75)

        # Cek apakah baris ini sudah pernah diisi
        cur_g = sample_df.at[current_idx, "gameplay"]
        cur_s = sample_df.at[current_idx, "story"]
        cur_t = sample_df.at[current_idx, "technical"]
        if pd.notnull(cur_g) and str(cur_g).strip() in VALID_LABELS:
            print(f"[Sudah terisi: Gameplay={cur_g}, Story={cur_s}, Tech={cur_t}]")

        user_input = input("Masukkan 3 angka (contoh '102' atau '1 0 2'): ").strip().lower()

        if user_input == "q":
            save_progress()
            print("\nProgres berhasil disimpan ke data/annotation/ground_truth.csv. Sampai jumpa!")
            return

        if user_input == "u":
            if history:
                current_idx = history.pop()
                continue
            else:
                input("\nTidak ada riwayat untuk di-undo. Tekan Enter untuk lanjut...")
                continue

        if user_input == "s":
            current_idx += 1
            continue

        # Parsing input 3 angka (mendukung format '102', '1 0 2', '1,0,2', '10 2')
        cleaned_chars = [c for c in user_input if c in {"0", "1", "2"}]

        if len(cleaned_chars) != 3:
            input("\nInput tidak valid! Harap masukkan tepat 3 angka (0, 1, atau 2). Tekan Enter...")
            continue

        g_val, s_val, t_val = int(cleaned_chars[0]), int(cleaned_chars[1]), int(cleaned_chars[2])

        sample_df.at[current_idx, "gameplay"] = g_val
        sample_df.at[current_idx, "story"] = s_val
        sample_df.at[current_idx, "technical"] = t_val

        save_progress()
        history.append(current_idx)
        current_idx += 1

    clear_screen()
    save_progress()
    print("=" * 75)
    print(" SELAMAT! SELURUH 1.500 ULASAN TELAH SELESAI DIANOTASI 100%!")
    print(" Berkas tersimpan di: data/annotation/ground_truth.csv")
    print(" Sekarang Anda siap menjalankan: 'python run.py split'")
    print("=" * 75)


if __name__ == "__main__":
    main()
