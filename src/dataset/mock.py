"""Mock dataset generator for pipeline testing and validation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetMock")


def generate_mock_ground_truth(
    output_path: str | Path = "data/annotation/ground_truth_test.csv",
    n_samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Menghasilkan dataset dummy ground truth berlabel untuk pengujian pipeline.

    Label aspek bernilai {0: none, 1: pos, 2: neg}.
    Memuat ulasan ekstrem (> 5.000 karakter dan multi-kalimat) untuk menguji ketahanan tokenisasi.

    Args:
        output_path: Path tujuan berkas CSV.
        n_samples: Jumlah baris data yang dihasilkan (default 100).
        seed: Random seed untuk determinisme.

    Returns:
        pd.DataFrame dengan kolom [review_id, app_id, clean_text, gameplay, story, technical].
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    # 1. Template ulasan normal
    templates = [
        "The gameplay is smooth and fluid, but the storyline feels shallow and repetitive.",
        "Stunning visuals and great story, though I experienced stuttering and technical performance drops.",
        "Combat system is responsive and addictive! Zero bugs encountered in my 50 hours of playing.",
        "Terrible port with constant crashes. The technical optimization is nonexistent.",
        "A peaceful narrative experience with touching characters and beautiful soundtrack.",
        "Challenging platforming mechanics, high difficulty curve, but very rewarding.",
        "Decent indie game. Story is average, controls are tight, runs at 60 fps smoothly.",
        "Boring mechanics and uninspired dialogue. Do not recommend unless on huge sale.",
    ]

    records = []
    # 2. Sisipkan 2 ulasan ekstrem di awal
    # Kasus ekstrem 1: Teks super panjang > 5.000 karakter
    long_essay = (
        "This indie game represents an intricate examination of pacing, design, and narrative depth. "
        "The mechanics intertwine seamlessly with level progression, offering multiple tactical approaches. "
    ) * 45  # ~7.000 karakter
    records.append(
        {
            "review_id": "mock_rev_000",
            "app_id": 504230,
            "clean_text": long_essay,
            "gameplay": 1,
            "story": 1,
            "technical": 1,
        }
    )

    # Kasus ekstrem 2: Teks multi-kalimat kompleks dengan tanda baca campuran
    multi_sentence = (
        "First 10 hours: pure joy! Next 20 hours: challenging puzzles and unexpected boss fights. "
        "Audio design is 10/10; voice acting is genuinely heartwarming. "
        "However, frame drops appear during stage 4 lava transitions. Overall, a solid 9/10."
    )
    records.append(
        {
            "review_id": "mock_rev_001",
            "app_id": 1145360,
            "clean_text": multi_sentence,
            "gameplay": 1,
            "story": 1,
            "technical": 2,
        }
    )

    # 3. Buat sisa baris hingga n_samples
    app_ids = [367520, 504230, 1145360, 588650, 413150]
    for i in range(2, n_samples):
        text = templates[i % len(templates)] + f" (Variant #{i})"
        records.append(
            {
                "review_id": f"mock_rev_{i:03d}",
                "app_id": int(np.random.choice(app_ids)),
                "clean_text": text,
                "gameplay": int(np.random.choice([0, 1, 2], p=[0.4, 0.35, 0.25])),
                "story": int(np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])),
                "technical": int(np.random.choice([0, 1, 2], p=[0.5, 0.2, 0.3])),
            }
        )

    df = pd.DataFrame(records, columns=["review_id", "app_id", "clean_text", "gameplay", "story", "technical"])
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    logger.info(
        "Mock ground truth berhasil dibuat: %d baris -> %s",
        len(df),
        output_file,
    )
    return df
