# app/services/csv_loader.py
"""
CSV 윈도우 로더: sampled_data.csv에서 연속된 N행을 시드 기반으로 재현 가능하게 샘플링
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import hashlib

# CSV 파일 경로
CSV_PATH = Path(__file__).parent.parent.parent.parent / "artifacts" / "sampled_data.csv"

# 필수 컬럼 (NASA 28개 변수)
REQUIRED_COLUMNS = [
    "ALLSKY_SFC_LW_DWN",
    "ALLSKY_SFC_PAR_TOT",
    "ALLSKY_SFC_SW_DIFF",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_UVA",
    "ALLSKY_SFC_UVB",
    "ALLSKY_SFC_UV_INDEX",
    "ALLSKY_SRF_ALB",
    "CLOUD_AMT",
    "CLRSKY_SFC_PAR_TOT",
    "CLRSKY_SFC_SW_DWN",
    "GWETPROF",
    "GWETROOT",
    "GWETTOP",
    "PRECTOTCORR",
    "PRECTOTCORR_SUM",
    "PS",
    "QV2M",
    "RH2M",
    "T2M",
    "T2MDEW",
    "T2MWET",
    "T2M_MAX",
    "T2M_MIN",
    "T2M_RANGE",
    "TOA_SW_DWN",
    "TS",
]


class CSVWindowLoader:
    """CSV에서 연속된 윈도우를 시드 기반으로 샘플링"""

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or CSV_PATH
        self._df_cache = None
        self._total_rows = None

    def _load_csv(self) -> pd.DataFrame:
        """CSV 파일 로드 및 검증"""
        if self._df_cache is not None:
            return self._df_cache

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {self.csv_path}")

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise RuntimeError(f"CSV 로드 실패: {e}")

        # 컬럼 검증
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")

        self._df_cache = df
        self._total_rows = len(df)
        return df

    def pick_window(self, seed: int, window: int = 6) -> Tuple[int, int]:
        """
        시드 기반으로 시작 인덱스를 결정하여 윈도우 범위 반환

        Args:
            seed: 난수 시드 (재현성 보장)
            window: 윈도우 크기 (기본 6)

        Returns:
            (start_row, end_row) 튜플 (inclusive)
        """
        df = self._load_csv()
        total_rows = len(df)

        if window > total_rows:
            raise ValueError(f"윈도우 크기({window})가 전체 행 수({total_rows})보다 큽니다")

        # 시드 기반 난수 생성
        rng = np.random.RandomState(seed)
        max_start = total_rows - window
        start_row = rng.randint(0, max_start + 1)
        end_row = start_row + window - 1

        return start_row, end_row

    def load_window(
        self,
        seed: int,
        window: int = 6,
        start_row: Optional[int] = None,
        end_row: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        윈도우 데이터 로드

        Args:
            seed: 난수 시드
            window: 윈도우 크기
            start_row: 시작 행 (지정 시 seed 무시)
            end_row: 종료 행 (start_row와 함께 지정 시 사용)

        Returns:
            윈도우 데이터프레임 (shape: [window, 28])
        """
        df = self._load_csv()

        # 명시적으로 start/end가 주어지지 않으면 시드로 결정
        if start_row is None or end_row is None:
            start_row, end_row = self.pick_window(seed, window)

        # 범위 검증
        if start_row < 0 or end_row >= len(df):
            raise ValueError(
                f"윈도우 범위가 유효하지 않습니다: [{start_row}, {end_row}], 전체 행 수: {len(df)}"
            )

        if end_row - start_row + 1 != window:
            raise ValueError(
                f"윈도우 크기 불일치: 요청={window}, 실제={(end_row - start_row + 1)}"
            )

        # 슬라이싱 (inclusive)
        window_df = df.iloc[start_row : end_row + 1][REQUIRED_COLUMNS].copy()

        # 결측치 처리 (forward fill → backward fill → 0)
        window_df = window_df.fillna(method="ffill").fillna(method="bfill").fillna(0)

        return window_df

    def get_total_rows(self) -> int:
        """전체 행 수 반환"""
        if self._total_rows is None:
            self._load_csv()
        return self._total_rows

    def get_window_hash(self, seed: int, window: int) -> str:
        """윈도우 고유 해시 생성 (캐시 키 용도)"""
        start_row, end_row = self.pick_window(seed, window)
        hash_input = f"{self.csv_path}:{start_row}:{end_row}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


# 싱글톤 인스턴스
_loader = None


def get_csv_loader() -> CSVWindowLoader:
    """글로벌 CSVWindowLoader 인스턴스 반환"""
    global _loader
    if _loader is None:
        _loader = CSVWindowLoader()
    return _loader


# 편의 함수
def load_climate_window(seed: int, window: int = 6) -> pd.DataFrame:
    """
    시드 기반으로 기후 데이터 윈도우 로드

    Args:
        seed: 난수 시드
        window: 윈도우 크기

    Returns:
        DataFrame (shape: [window, 28])
    """
    loader = get_csv_loader()
    return loader.load_window(seed, window)


def get_window_statistics(df: pd.DataFrame) -> dict:
    """
    윈도우 데이터의 기본 통계 반환

    Args:
        df: 윈도우 데이터프레임

    Returns:
        통계 딕셔너리 (mean, std, min, max)
    """
    stats = {}
    for col in df.columns:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
    return stats


def compare_windows(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """
    델타 적용 전후 윈도우 비교

    Args:
        df_before: 원본 윈도우
        df_after: 델타 적용 후 윈도우

    Returns:
        변화량 딕셔너리
    """
    changes = {}
    for col in df_before.columns:
        before_mean = df_before[col].mean()
        after_mean = df_after[col].mean()
        changes[col] = {
            "before_mean": float(before_mean),
            "after_mean": float(after_mean),
            "delta": float(after_mean - before_mean),
            "pct_change": float((after_mean - before_mean) / (before_mean + 1e-9) * 100),
        }
    return changes
