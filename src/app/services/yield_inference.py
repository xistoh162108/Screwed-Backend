import os, json, joblib
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb

class YieldPredictorLGBM:
    def __init__(self, artifacts_dir: str):
        """
        artifacts_dir 내부에 다음이 있어야 함:
          - LightGBM_PpA_model.pkl  또는  model.pt  또는  model.txt
          - feature_names.json
        """
        self.artifacts_dir = os.path.abspath(artifacts_dir)
        self.pkl_path = os.path.join(self.artifacts_dir, "LightGBM_PpA_model.pkl")
        self.pt_path  = os.path.join(self.artifacts_dir, "model.pt")
        self.txt_path = os.path.join(self.artifacts_dir, "model.txt")
        self.features_path = os.path.join(self.artifacts_dir, "feature_names.json")

        # 1) feature_names.json
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"feature_names.json not found: {self.features_path}")
        with open(self.features_path, "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)
        if not isinstance(self.feature_names, list) or not self.feature_names:
            raise ValueError("feature_names.json must be a non-empty list.")

        # 2) 모델 로드 (여러 형식 지원)
        bundle = None
        model_path_used = None
        if os.path.exists(self.pkl_path):
            bundle = joblib.load(self.pkl_path)
            model_path_used = self.pkl_path
        elif os.path.exists(self.pt_path):
            bundle = torch.load(self.pt_path, map_location="cpu")
            model_path_used = self.pt_path
        elif os.path.exists(self.txt_path):
            # LightGBM native text model
            self.imputer = None
            self.booster = lgb.Booster(model_file=self.txt_path)
            self.best_iteration = getattr(self.booster, "best_iteration", None) or None
            return
        else:
            raise FileNotFoundError(
                "Model bundle not found. Expected one of:\n"
                f"  - {self.pkl_path}\n"
                f"  - {self.pt_path}\n"
                f"  - {self.txt_path}"
            )

        # 3) bundle 해석
        self.imputer = None
        self.booster = None
        self.best_iteration = None

        if isinstance(bundle, dict):
            # 정석 번들: {"imputer": ..., "booster": ..., "best_iteration": ...}
            self.imputer = bundle.get("imputer", None)
            self.booster = bundle.get("booster", None)
            self.best_iteration = bundle.get("best_iteration", None)
        elif isinstance(bundle, lgb.Booster):
            self.booster = bundle
        else:
            # 혹시 피클/pt가 Booster 단독 아닌 다른 형태면 시도 후 실패
            raise TypeError(
                f"Unsupported model bundle type from {model_path_used}: {type(bundle)}.\n"
                "Expected dict with keys or lgb.Booster or model.txt."
            )

        # 4) best_iteration 보정 (없으면 전체 트리 사용)
        try:
            if self.best_iteration is None or int(self.best_iteration) <= 0:
                self.best_iteration = int(getattr(self.booster, "best_iteration", 0) or 0) or None
        except Exception:
            self.best_iteration = None

        # 5) 최종 검증: booster는 필수, imputer는 선택
        if self.booster is None:
            raise ValueError("Model bundle missing 'booster'.")

    # --------- helpers ---------
    def _ensure_dataframe(self, data) -> pd.DataFrame:
        """
        data: DataFrame / list[dict] / dict / ndarray
        반환: feature_names 순서로 정렬된 DataFrame
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, (list, tuple)) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            arr = data
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(f"ndarray must have shape (*, {len(self.feature_names)})")
            df = pd.DataFrame(arr, columns=self.feature_names)
        else:
            raise TypeError("Unsupported data type for prediction.")

        # 누락 컬럼 채우기(0) + 여분 드랍 + 순서 고정
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_names]
        return df

    # --------- public API ---------
    def predict(self, data, return_with_inputs: bool = False) -> pd.DataFrame:
        df = self._ensure_dataframe(data)
        X = df.values
        if self.imputer is not None:
            X = self.imputer.transform(X)
        y_pred = self.booster.predict(X, num_iteration=self.best_iteration)
        out = pd.DataFrame({"PpA_pred": y_pred})
        return pd.concat([df.reset_index(drop=True), out], axis=1) if return_with_inputs else out

    def predict_one(self, row_dict: dict) -> float:
        df = self._ensure_dataframe(row_dict)
        X = df.values
        if self.imputer is not None:
            X = self.imputer.transform(X)
        y_pred = self.booster.predict(X, num_iteration=self.best_iteration)
        return float(y_pred[0])


class MultiCropYieldService:
    """곡물별 아티팩트 디렉토리를 매핑해 멀티 모델 핸들링."""
    def __init__(self, root_output_dir: str):
        # 예: /…/artifacts/yieldInference/{crop}
        base = os.path.abspath(root_output_dir)
        self.dirs = {
            "MAIZE":   os.path.join(base, "maize"),
            "RICE":    os.path.join(base, "rice"),
            "SOYBEAN": os.path.join(base, "soybean"),
            "WHEAT":   os.path.join(base, "wheat"),
        }
        self.models = {}

    def get(self, crop: str) -> YieldPredictorLGBM:
        key = crop.strip().upper()
        if key not in self.dirs:
            raise ValueError(f"Unsupported crop: {crop}")
        if key not in self.models:
            self.models[key] = YieldPredictorLGBM(self.dirs[key])
        return self.models[key]