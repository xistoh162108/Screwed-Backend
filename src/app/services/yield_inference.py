import os, json, glob, joblib
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb

class MAIZE_PredictorLGBM:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = os.path.abspath(artifacts_dir)

        # ---- 1) candidate files (이름 유연화) ----
        cand_pt  = sorted(glob.glob(os.path.join(self.artifacts_dir, "*.pt")))
        cand_pkl = sorted(glob.glob(os.path.join(self.artifacts_dir, "*.pkl")))
        cand_txt = sorted(glob.glob(os.path.join(self.artifacts_dir, "*.txt")))
        self.features_path = os.path.join(self.artifacts_dir, "feature_names.json")

        # ---- 2) feature_names 로드 (없으면 대체 전략) ----
        self.feature_names = None
        if os.path.exists(self.features_path):
            with open(self.features_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
            if not isinstance(self.feature_names, list) or not self.feature_names:
                raise ValueError("feature_names.json must be a non-empty list.")

        # ---- 3) 모델 로드 (여러 형식/직렬화 처리) ----
        self.imputer = None
        self.booster = None
        self.best_iteration = None

        model_loaded = False
        load_errors = []

        # helper: 번들 해석
        def _consume_bundle(bundle):
            nonlocal model_loaded
            if isinstance(bundle, dict):
                # 가장 일반적인 번들 형태
                self.imputer = bundle.get("imputer", None)
                self.booster = bundle.get("booster", None)
                self.best_iteration = bundle.get("best_iteration", None)
                if self.booster is None:
                    raise TypeError("dict bundle has no 'booster'.")
                model_loaded = True
            elif isinstance(bundle, lgb.Booster):
                self.booster = bundle
                model_loaded = True
            else:
                # PyTorch 모형이면 여기서 중단 (이 클래스는 LightGBM 추론기)
                import types
                torch_module_types = (torch.nn.Module, )
                if isinstance(bundle, torch.jit.ScriptModule) or isinstance(bundle, torch_module_types):
                    raise TypeError(
                        "Loaded a PyTorch model (.pt). "
                        "This predictor is for LightGBM. "
                        "Use a PyTorch predictor class instead."
                    )
                raise TypeError(f"Unsupported model object type: {type(bundle)}")

        # (a) .txt → native LightGBM
        for p in cand_txt:
            try:
                self.booster = lgb.Booster(model_file=p)
                model_loaded = True
                break
            except Exception as e:
                load_errors.append((p, str(e)))

        # (b) .pt / .pkl → torch.load or joblib.load
        if not model_loaded:
            for p in cand_pt + cand_pkl:
                # torch.load 먼저, 실패하면 joblib.load
                bundle = None
                tried = []
                try:
                    bundle = torch.load(p, map_location="cpu")
                    tried.append("torch.load")
                except Exception as e:
                    load_errors.append((p, f"torch.load: {e}"))
                if bundle is None:
                    try:
                        bundle = joblib.load(p)
                        tried.append("joblib.load")
                    except Exception as e:
                        load_errors.append((p, f"joblib.load: {e}"))
                if bundle is not None:
                    try:
                        _consume_bundle(bundle)
                        break
                    except Exception as e:
                        load_errors.append((p, f"bundle-consume: {e}"))

        if not model_loaded:
            msgs = "\n".join([f"- {pp}: {ee}" for pp, ee in load_errors[-6:]])
            raise FileNotFoundError(
                "No usable model found under artifacts_dir. Tried *.txt, *.pt, *.pkl.\n"
                + msgs
            )

        # ---- 4) best_iteration 보정 ----
        try:
            if self.best_iteration is None or int(self.best_iteration) <= 0:
                self.best_iteration = int(getattr(self.booster, "best_iteration", 0) or 0) or None
        except Exception:
            self.best_iteration = None

        # ---- 5) feature_names 보완 (없을 때 Booster에서 꺼내기) ----
        if self.feature_names is None:
            try:
                self.feature_names = list(self.booster.feature_name())
            except Exception:
                # 마지막 대안: 추론 시 들어오는 DF의 컬럼을 그대로 사용하도록 둠
                # (predict에서 None이면 DF.columns 사용)
                self.feature_names = None

    # ---------- helpers ----------
    def _ensure_dataframe(self, data) -> pd.DataFrame:
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
            if self.feature_names is None:
                raise ValueError("feature_names is unknown. Provide a DataFrame with named columns.")
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(f"ndarray must have shape (*, {len(self.feature_names)})")
            df = pd.DataFrame(arr, columns=self.feature_names)
        else:
            raise TypeError("Unsupported data type for prediction.")

        # feature_names가 있으면 그 순서로 정렬/보정
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_names]
        return df

    # ---------- public API ----------
    def predict(self, data, return_with_inputs: bool = False) -> pd.DataFrame:
        df = self._ensure_dataframe(data)
        df = df.set_index(pd.Index(range(len(df))))

        target_crop = 'MAIZE'

        climate_cols = [
            'ALLSKY_SFC_LW_DWN','ALLSKY_SFC_PAR_TOT','ALLSKY_SFC_SW_DIFF','ALLSKY_SFC_SW_DNI',
            'ALLSKY_SFC_SW_DWN','ALLSKY_SFC_UVA','ALLSKY_SFC_UVB','ALLSKY_SFC_UV_INDEX',
            'ALLSKY_SRF_ALB','CLOUD_AMT','CLRSKY_SFC_PAR_TOT','CLRSKY_SFC_SW_DWN',
            'GWETPROF','GWETROOT','GWETTOP','PRECTOTCORR','PRECTOTCORR_SUM','PS','QV2M',
            'RH2M','T2M','T2MDEW','T2MWET','T2M_MAX','T2M_MIN','T2M_RANGE','TOA_SW_DWN','TS'
        ]

        df = df[climate_cols]

        wide = df.stack().unstack(level=0)

        wide.columns = [f"{var}_{idx}" for var, idx in wide.columns]
        X = wide.reset_index()

        if getattr(self, "imputer", None) is not None:
            X = self.imputer.transform(X)
        y_pred = self.booster.predict(X, num_iteration=self.best_iteration)
        out = pd.DataFrame({"PpA_pred": y_pred})
        return pd.concat([df.reset_index(drop=True), out], axis=1) if return_with_inputs else out

    def predict_one(self, row_dict: dict) -> float:
        df = self._ensure_dataframe(row_dict)
        X = df.values
        if getattr(self, "imputer", None) is not None:
            X = self.imputer.transform(X)
        y_pred = self.booster.predict(X, num_iteration=self.best_iteration)
        return float(y_pred[0])

class WHEAT_PredictorLGBM:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = os.path.abspath(artifacts_dir)

        # ---- 1) candidate files (이름 유연화) ----
        cand_pt  = sorted(glob.glob(os.path.join(self.artifacts_dir, "*.pt")))
        cand_pkl = sorted(glob.glob(os.path.join(self.artifacts_dir, "*.pkl")))
        cand_txt = sorted(glob.glob(os.path.join(self.artifacts_dir, "*.txt")))
        self.features_path = os.path.join(self.artifacts_dir, "feature_names.json")

        # ---- 2) feature_names 로드 (없으면 대체 전략) ----
        self.feature_names = None
        if os.path.exists(self.features_path):
            with open(self.features_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
            if not isinstance(self.feature_names, list) or not self.feature_names:
                raise ValueError("feature_names.json must be a non-empty list.")

        # ---- 3) 모델 로드 (여러 형식/직렬화 처리) ----
        self.imputer = None
        self.booster = None
        self.best_iteration = None

        model_loaded = False
        load_errors = []

        # helper: 번들 해석
        def _consume_bundle(bundle):
            nonlocal model_loaded
            if isinstance(bundle, dict):
                # 가장 일반적인 번들 형태
                self.imputer = bundle.get("imputer", None)
                self.booster = bundle.get("booster", None)
                self.best_iteration = bundle.get("best_iteration", None)
                if self.booster is None:
                    raise TypeError("dict bundle has no 'booster'.")
                model_loaded = True
            elif isinstance(bundle, lgb.Booster):
                self.booster = bundle
                model_loaded = True
            else:
                # PyTorch 모형이면 여기서 중단 (이 클래스는 LightGBM 추론기)
                import types
                torch_module_types = (torch.nn.Module, )
                if isinstance(bundle, torch.jit.ScriptModule) or isinstance(bundle, torch_module_types):
                    raise TypeError(
                        "Loaded a PyTorch model (.pt). "
                        "This predictor is for LightGBM. "
                        "Use a PyTorch predictor class instead."
                    )
                raise TypeError(f"Unsupported model object type: {type(bundle)}")

        # (a) .txt → native LightGBM
        for p in cand_txt:
            try:
                self.booster = lgb.Booster(model_file=p)
                model_loaded = True
                break
            except Exception as e:
                load_errors.append((p, str(e)))

        # (b) .pt / .pkl → torch.load or joblib.load
        if not model_loaded:
            for p in cand_pt + cand_pkl:
                # torch.load 먼저, 실패하면 joblib.load
                bundle = None
                tried = []
                try:
                    bundle = torch.load(p, map_location="cpu")
                    tried.append("torch.load")
                except Exception as e:
                    load_errors.append((p, f"torch.load: {e}"))
                if bundle is None:
                    try:
                        bundle = joblib.load(p)
                        tried.append("joblib.load")
                    except Exception as e:
                        load_errors.append((p, f"joblib.load: {e}"))
                if bundle is not None:
                    try:
                        _consume_bundle(bundle)
                        break
                    except Exception as e:
                        load_errors.append((p, f"bundle-consume: {e}"))

        if not model_loaded:
            msgs = "\n".join([f"- {pp}: {ee}" for pp, ee in load_errors[-6:]])
            raise FileNotFoundError(
                "No usable model found under artifacts_dir. Tried *.txt, *.pt, *.pkl.\n"
                + msgs
            )

        # ---- 4) best_iteration 보정 ----
        try:
            if self.best_iteration is None or int(self.best_iteration) <= 0:
                self.best_iteration = int(getattr(self.booster, "best_iteration", 0) or 0) or None
        except Exception:
            self.best_iteration = None

        # ---- 5) feature_names 보완 (없을 때 Booster에서 꺼내기) ----
        if self.feature_names is None:
            try:
                self.feature_names = list(self.booster.feature_name())
            except Exception:
                # 마지막 대안: 추론 시 들어오는 DF의 컬럼을 그대로 사용하도록 둠
                # (predict에서 None이면 DF.columns 사용)
                self.feature_names = None

    # ---------- helpers ----------
    def _ensure_dataframe(self, data) -> pd.DataFrame:
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
            if self.feature_names is None:
                raise ValueError("feature_names is unknown. Provide a DataFrame with named columns.")
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(f"ndarray must have shape (*, {len(self.feature_names)})")
            df = pd.DataFrame(arr, columns=self.feature_names)
        else:
            raise TypeError("Unsupported data type for prediction.")

        # feature_names가 있으면 그 순서로 정렬/보정
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_names]
        return df

    # ---------- public API ----------
    def predict(self, data, return_with_inputs: bool = False) -> pd.DataFrame:
        df = self._ensure_dataframe(data)
        df = df.set_index(pd.Index(range(len(df))))

        target_crop = 'MAIZE'

        climate_cols = [
            'ALLSKY_SFC_LW_DWN','ALLSKY_SFC_PAR_TOT','ALLSKY_SFC_SW_DIFF','ALLSKY_SFC_SW_DNI',
            'ALLSKY_SFC_SW_DWN','ALLSKY_SFC_UVA','ALLSKY_SFC_UVB','ALLSKY_SFC_UV_INDEX',
            'ALLSKY_SRF_ALB','CLOUD_AMT','CLRSKY_SFC_PAR_TOT','CLRSKY_SFC_SW_DWN',
            'GWETPROF','GWETROOT','GWETTOP','PRECTOTCORR','PRECTOTCORR_SUM','PS','QV2M',
            'RH2M','T2M','T2MDEW','T2MWET','T2M_MAX','T2M_MIN','T2M_RANGE','TOA_SW_DWN','TS'
        ]

        df = df[climate_cols]

        wide = df.stack().unstack(level=0)

        wide.columns = [f"{var}_{idx}" for var, idx in wide.columns]
        X = wide.reset_index()

        if getattr(self, "imputer", None) is not None:
            X = self.imputer.transform(X)
        y_pred = self.booster.predict(X, num_iteration=self.best_iteration)
        out = pd.DataFrame({"PpA_pred": y_pred})
        return pd.concat([df.reset_index(drop=True), out], axis=1) if return_with_inputs else out

    def predict_one(self, row_dict: dict) -> float:
        df = self._ensure_dataframe(row_dict)
        X = df.values
        if getattr(self, "imputer", None) is not None:
            X = self.imputer.transform(X)
        y_pred = self.booster.predict(X, num_iteration=self.best_iteration)
        return float(y_pred[0])

class RICE_PredictorLGBM:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = os.path.abspath(artifacts_dir)

        # ---- 1) candidate files (이름 유연화) ----
        cand_pt  = sorted(glob.glob(os.path.join(self.artifacts_dir, "*.pt")))
        cand_pkl = sorted(glob.glob(os.path.join(self.artifacts_dir, "*.pkl")))
        cand_txt = sorted(glob.glob(os.path.join(self.artifacts_dir, "*.txt")))
        self.features_path = os.path.join(self.artifacts_dir, "feature_names.json")

        # ---- 2) feature_names 로드 (없으면 대체 전략) ----
        self.feature_names = None
        if os.path.exists(self.features_path):
            with open(self.features_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
            if not isinstance(self.feature_names, list) or not self.feature_names:
                raise ValueError("feature_names.json must be a non-empty list.")

        # ---- 3) 모델 로드 (여러 형식/직렬화 처리) ----
        self.imputer = None
        self.booster = None
        self.best_iteration = None

        model_loaded = False
        load_errors = []

        # helper: 번들 해석
        def _consume_bundle(bundle):
            nonlocal model_loaded
            if isinstance(bundle, dict):
                # 가장 일반적인 번들 형태
                self.imputer = bundle.get("imputer", None)
                self.booster = bundle.get("booster", None)
                self.best_iteration = bundle.get("best_iteration", None)
                if self.booster is None:
                    raise TypeError("dict bundle has no 'booster'.")
                model_loaded = True
            elif isinstance(bundle, lgb.Booster):
                self.booster = bundle
                model_loaded = True
            else:
                # PyTorch 모형이면 여기서 중단 (이 클래스는 LightGBM 추론기)
                import types
                torch_module_types = (torch.nn.Module, )
                if isinstance(bundle, torch.jit.ScriptModule) or isinstance(bundle, torch_module_types):
                    raise TypeError(
                        "Loaded a PyTorch model (.pt). "
                        "This predictor is for LightGBM. "
                        "Use a PyTorch predictor class instead."
                    )
                raise TypeError(f"Unsupported model object type: {type(bundle)}")

        # (a) .txt → native LightGBM
        for p in cand_txt:
            try:
                self.booster = lgb.Booster(model_file=p)
                model_loaded = True
                break
            except Exception as e:
                load_errors.append((p, str(e)))

        # (b) .pt / .pkl → torch.load or joblib.load
        if not model_loaded:
            for p in cand_pt + cand_pkl:
                # torch.load 먼저, 실패하면 joblib.load
                bundle = None
                tried = []
                try:
                    bundle = torch.load(p, map_location="cpu")
                    tried.append("torch.load")
                except Exception as e:
                    load_errors.append((p, f"torch.load: {e}"))
                if bundle is None:
                    try:
                        bundle = joblib.load(p)
                        tried.append("joblib.load")
                    except Exception as e:
                        load_errors.append((p, f"joblib.load: {e}"))
                if bundle is not None:
                    try:
                        _consume_bundle(bundle)
                        break
                    except Exception as e:
                        load_errors.append((p, f"bundle-consume: {e}"))

        if not model_loaded:
            msgs = "\n".join([f"- {pp}: {ee}" for pp, ee in load_errors[-6:]])
            raise FileNotFoundError(
                "No usable model found under artifacts_dir. Tried *.txt, *.pt, *.pkl.\n"
                + msgs
            )

        # ---- 4) best_iteration 보정 ----
        try:
            if self.best_iteration is None or int(self.best_iteration) <= 0:
                self.best_iteration = int(getattr(self.booster, "best_iteration", 0) or 0) or None
        except Exception:
            self.best_iteration = None

        # ---- 5) feature_names 보완 (없을 때 Booster에서 꺼내기) ----
        if self.feature_names is None:
            try:
                self.feature_names = list(self.booster.feature_name())
            except Exception:
                # 마지막 대안: 추론 시 들어오는 DF의 컬럼을 그대로 사용하도록 둠
                # (predict에서 None이면 DF.columns 사용)
                self.feature_names = None

    # ---------- helpers ----------
    def _ensure_dataframe(self, data) -> pd.DataFrame:
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
            if self.feature_names is None:
                raise ValueError("feature_names is unknown. Provide a DataFrame with named columns.")
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(f"ndarray must have shape (*, {len(self.feature_names)})")
            df = pd.DataFrame(arr, columns=self.feature_names)
        else:
            raise TypeError("Unsupported data type for prediction.")

        # feature_names가 있으면 그 순서로 정렬/보정
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_names]
        return df

    # ---------- public API ----------
    def predict(self, data, return_with_inputs: bool = False) -> pd.DataFrame:
        df = self._ensure_dataframe(data)
        df = df.set_index(pd.Index(range(len(df))))

        target_crop = 'RICE'

        climate_cols = [
            'ALLSKY_SFC_LW_DWN','ALLSKY_SFC_PAR_TOT','ALLSKY_SFC_SW_DIFF','ALLSKY_SFC_SW_DNI',
            'ALLSKY_SFC_SW_DWN','ALLSKY_SFC_UVA','ALLSKY_SFC_UVB','ALLSKY_SFC_UV_INDEX',
            'ALLSKY_SRF_ALB','CLOUD_AMT','CLRSKY_SFC_PAR_TOT','CLRSKY_SFC_SW_DWN',
            'GWETPROF','GWETROOT','GWETTOP','PRECTOTCORR','PRECTOTCORR_SUM','PS','QV2M',
            'RH2M','T2M','T2MDEW','T2MWET','T2M_MAX','T2M_MIN','T2M_RANGE','TOA_SW_DWN','TS'
        ]

        df = df[climate_cols]

        wide = df.stack().unstack(level=0)

        wide.columns = [f"{var}_{idx}" for var, idx in wide.columns]
        X = wide.reset_index()

        if getattr(self, "imputer", None) is not None:
            X = self.imputer.transform(X)
        y_pred = self.booster.predict(X, num_iteration=self.best_iteration)
        out = pd.DataFrame({"PpA_pred": y_pred})
        return pd.concat([df.reset_index(drop=True), out], axis=1) if return_with_inputs else out

    def predict_one(self, row_dict: dict) -> float:
        df = self._ensure_dataframe(row_dict)
        X = df.values
        if getattr(self, "imputer", None) is not None:
            X = self.imputer.transform(X)
        y_pred = self.booster.predict(X, num_iteration=self.best_iteration)
        return float(y_pred[0])

class SOYBEAN_PredictorLGBM:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = os.path.abspath(artifacts_dir)

        # ---- 1) candidate files (이름 유연화) ----
        cand_pt  = sorted(glob.glob(os.path.join(self.artifacts_dir, "*.pt")))
        cand_pkl = sorted(glob.glob(os.path.join(self.artifacts_dir, "*.pkl")))
        cand_txt = sorted(glob.glob(os.path.join(self.artifacts_dir, "*.txt")))
        self.features_path = os.path.join(self.artifacts_dir, "feature_names.json")

        # ---- 2) feature_names 로드 (없으면 대체 전략) ----
        self.feature_names = None
        if os.path.exists(self.features_path):
            with open(self.features_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
            if not isinstance(self.feature_names, list) or not self.feature_names:
                raise ValueError("feature_names.json must be a non-empty list.")

        # ---- 3) 모델 로드 (여러 형식/직렬화 처리) ----
        self.imputer = None
        self.booster = None
        self.best_iteration = None

        model_loaded = False
        load_errors = []

        # helper: 번들 해석
        def _consume_bundle(bundle):
            nonlocal model_loaded
            if isinstance(bundle, dict):
                # 가장 일반적인 번들 형태
                self.imputer = bundle.get("imputer", None)
                self.booster = bundle.get("booster", None)
                self.best_iteration = bundle.get("best_iteration", None)
                if self.booster is None:
                    raise TypeError("dict bundle has no 'booster'.")
                model_loaded = True
            elif isinstance(bundle, lgb.Booster):
                self.booster = bundle
                model_loaded = True
            else:
                # PyTorch 모형이면 여기서 중단 (이 클래스는 LightGBM 추론기)
                import types
                torch_module_types = (torch.nn.Module, )
                if isinstance(bundle, torch.jit.ScriptModule) or isinstance(bundle, torch_module_types):
                    raise TypeError(
                        "Loaded a PyTorch model (.pt). "
                        "This predictor is for LightGBM. "
                        "Use a PyTorch predictor class instead."
                    )
                raise TypeError(f"Unsupported model object type: {type(bundle)}")

        # (a) .txt → native LightGBM
        for p in cand_txt:
            try:
                self.booster = lgb.Booster(model_file=p)
                model_loaded = True
                break
            except Exception as e:
                load_errors.append((p, str(e)))

        # (b) .pt / .pkl → torch.load or joblib.load
        if not model_loaded:
            for p in cand_pt + cand_pkl:
                # torch.load 먼저, 실패하면 joblib.load
                bundle = None
                tried = []
                try:
                    bundle = torch.load(p, map_location="cpu")
                    tried.append("torch.load")
                except Exception as e:
                    load_errors.append((p, f"torch.load: {e}"))
                if bundle is None:
                    try:
                        bundle = joblib.load(p)
                        tried.append("joblib.load")
                    except Exception as e:
                        load_errors.append((p, f"joblib.load: {e}"))
                if bundle is not None:
                    try:
                        _consume_bundle(bundle)
                        break
                    except Exception as e:
                        load_errors.append((p, f"bundle-consume: {e}"))

        if not model_loaded:
            msgs = "\n".join([f"- {pp}: {ee}" for pp, ee in load_errors[-6:]])
            raise FileNotFoundError(
                "No usable model found under artifacts_dir. Tried *.txt, *.pt, *.pkl.\n"
                + msgs
            )

        # ---- 4) best_iteration 보정 ----
        try:
            if self.best_iteration is None or int(self.best_iteration) <= 0:
                self.best_iteration = int(getattr(self.booster, "best_iteration", 0) or 0) or None
        except Exception:
            self.best_iteration = None

        # ---- 5) feature_names 보완 (없을 때 Booster에서 꺼내기) ----
        if self.feature_names is None:
            try:
                self.feature_names = list(self.booster.feature_name())
            except Exception:
                # 마지막 대안: 추론 시 들어오는 DF의 컬럼을 그대로 사용하도록 둠
                # (predict에서 None이면 DF.columns 사용)
                self.feature_names = None

    # ---------- helpers ----------
    def _ensure_dataframe(self, data) -> pd.DataFrame:
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
            if self.feature_names is None:
                raise ValueError("feature_names is unknown. Provide a DataFrame with named columns.")
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(f"ndarray must have shape (*, {len(self.feature_names)})")
            df = pd.DataFrame(arr, columns=self.feature_names)
        else:
            raise TypeError("Unsupported data type for prediction.")

        # feature_names가 있으면 그 순서로 정렬/보정
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_names]
        return df

    # ---------- public API ----------
    def predict(self, data, return_with_inputs: bool = False) -> pd.DataFrame:
        df = self._ensure_dataframe(data)
        df = df.set_index(pd.Index(range(len(df))))

        target_crop = 'SOYBEAN'

        climate_cols = [
            'ALLSKY_SFC_LW_DWN','ALLSKY_SFC_PAR_TOT','ALLSKY_SFC_SW_DIFF','ALLSKY_SFC_SW_DNI',
            'ALLSKY_SFC_SW_DWN','ALLSKY_SFC_UVA','ALLSKY_SFC_UVB','ALLSKY_SFC_UV_INDEX',
            'ALLSKY_SRF_ALB','CLOUD_AMT','CLRSKY_SFC_PAR_TOT','CLRSKY_SFC_SW_DWN',
            'GWETPROF','GWETROOT','GWETTOP','PRECTOTCORR','PRECTOTCORR_SUM','PS','QV2M',
            'RH2M','T2M','T2MDEW','T2MWET','T2M_MAX','T2M_MIN','T2M_RANGE','TOA_SW_DWN','TS'
        ]

        df = df[climate_cols]

        wide = df.stack().unstack(level=0)

        wide.columns = [f"{var}_{idx}" for var, idx in wide.columns]
        X = wide.reset_index()

        if getattr(self, "imputer", None) is not None:
            X = self.imputer.transform(X)
        y_pred = self.booster.predict(X, num_iteration=self.best_iteration)
        out = pd.DataFrame({"PpA_pred": y_pred})
        return pd.concat([df.reset_index(drop=True), out], axis=1) if return_with_inputs else out

    def predict_one(self, row_dict: dict) -> float:
        df = self._ensure_dataframe(row_dict)
        X = df.values
        if getattr(self, "imputer", None) is not None:
            X = self.imputer.transform(X)
        y_pred = self.booster.predict(X, num_iteration=self.best_iteration)
        return float(y_pred[0])
