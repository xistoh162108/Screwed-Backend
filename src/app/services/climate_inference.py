# -*- coding: utf-8 -*-
"""
CPU 전용 추론 서비스
- 모델: FNN + ResLSTM + Attention Head (gaussian 헤드)
- 입력: 최근 L(기본 6)개 타임스텝의 (F) 피처 + (month, lat, lon)
- 보정: calibration.json (per-feature log_t_f, scalar_s) 적용
- 정규화: normalizer_stats.json (global or groupless)

필수 파일 (기본 경로: ./artifacts):
  - model.pt
  - normalizer_stats.json
  - calibration.json
"""

import os, json, math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# =============================
# 환경/기본 설정 (CPU 고정)
# =============================
DEVICE = torch.device("cpu")
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# =============================
# 모델 정의 (학습 때와 동일)
# =============================
class ResLSTMBlock(nn.Module):
    def __init__(self, d_in, d_out, layer_norm=True, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_in, hidden_size=d_out, batch_first=True)
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else None
        self.act  = nn.GELU()
        self.ln   = nn.LayerNorm(d_out) if layer_norm else nn.Identity()
        self.do   = nn.Dropout(dropout)
        # init
        for name, p in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x):
        y, _ = self.lstm(x)
        res = x if self.proj is None else self.proj(x)
        y = self.do(self.ln(y + res))
        y = self.act(y)
        return y


class FNN_ResLSTM_AttnHead(nn.Module):
    """
    Pre-FNN → ResLSTM×L → Attention pooling → Head
    Gaussian 헤드: out=[mu, logvar] (size 2F)
    """
    def __init__(self, in_dim, out_dim,
                 pre_hidden=256, pre_out=128,
                 lstm_hidden=256, num_layers=3,
                 post_hidden=128, dropout=0.15, layer_norm=True):
        super().__init__()
        self.F = out_dim

        self.pre = nn.Sequential(
            nn.Linear(in_dim, pre_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pre_hidden, pre_out),
            nn.GELU(),
        )
        self.pre_ln = nn.LayerNorm(pre_out) if layer_norm else nn.Identity()

        blocks, d_in = [], pre_out
        for _ in range(num_layers):
            blocks.append(ResLSTMBlock(d_in, lstm_hidden, layer_norm=layer_norm, dropout=dropout))
            d_in = lstm_hidden
        self.blocks = nn.Sequential(*blocks)

        self.attn = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, post_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(post_hidden, out_dim * 2),   # [mu, logvar]
        )

    def forward(self, x):  # x: (B,L,in_dim)
        z = self.pre_ln(self.pre(x))
        z = self.blocks(z)                    # (B,L,H)
        score = self.attn(z).squeeze(-1)     # (B,L)
        w = torch.softmax(score, dim=1).unsqueeze(-1)  # (B,L,1)
        ctx = torch.sum(w * z, dim=1)        # (B,H)
        out = self.head(ctx)                 # (B, 2F)
        mu, logvar = out[..., :self.F], out[..., self.F:]
        return mu, logvar


# =============================
# 정규화/역정규화
# =============================
class Normalizer:
    """
    normalizer_stats.json의 구조:
      {
        "__global__": { "<feat>": {...}, ... },
        "ARG|-38.0|-68.75": { "<feat>": {...}, ... },
        ...
      }
    기본은 __global__를 사용하고, 필요하면 select_group(code, lat, lon)으로 그룹 스탯으로 전환.
    """
    def __init__(self, stats_dict: dict, feature_names: Optional[List[str]] = None):
        if "__global__" not in stats_dict:
            raise ValueError("normalizer_stats.json 최상위에 '__global__' 키가 필요합니다.")
        self.stats_root = stats_dict
        self.S = stats_dict["__global__"]  # 현재 활성 스탯(초기값: 글로벌)
        self.feature_names = list(feature_names) if feature_names is not None else list(self.S.keys())

    def select_group(self, code: str = None, lat: float = None, lon: float = None):
        """정확히 일치하는 그룹키가 있으면 그걸로 스위치, 없으면 글로벌 유지."""
        if code is None or lat is None or lon is None:
            self.S = self.stats_root["__global__"]
            return

        # 저장 시 문자열 포맷을 그대로 맞춰야 함.
        # 예: "ARG|-38.0|-68.75" (lat은 소수 1자리, lon은 소수 2~3자리로 저장되었을 수 있음)
        # 가장 보편적으로 쓰이는 포맷 몇 개를 시도해봅니다.
        candidates = [
            f"{code}|{lat:.1f}|{lon:.2f}",
            f"{code}|{lat:.1f}|{lon:.3f}",
            f"{code}|{lat:.1f}|{lon}",      # lon 원본 그대로
            f"{code}|{lat}|{lon}",          # 둘 다 그대로
        ]
        for k in candidates:
            if k in self.stats_root:
                self.S = self.stats_root[k]
                return
        # 못 찾으면 글로벌
        self.S = self.stats_root["__global__"]

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (..., F)
        X = X.clone()
        eps = 1e-6
        F = X.shape[-1]
        # self.S는 현재 활성(글로벌 또는 그룹)의 피처-스탯 딕션어리
        for fi, f in enumerate(self.feature_names):
            st = self.S[f]  # ← 여기서 KeyError 났던 부분이 이제 정상 동작
            typ = st['type']
            if typ == 'z':
                X[..., fi] = (X[..., fi] - st['mean']) / max(st['std'], eps)
            elif typ == 'log1p_z':
                X[..., fi] = (torch.log1p(torch.clamp(X[..., fi], min=0.0)) - st['mean']) / max(st['std'], eps)
            elif typ == 'sqrt_z':
                X[..., fi] = (torch.sqrt(torch.clamp(X[..., fi], min=0.0)) - st['mean']) / max(st['std'], eps)
            elif typ == 'logit_z':
                xmin, xmax = st['xmin'], st['xmax']
                rng = max(xmax - xmin, eps)
                x01 = (X[..., fi] - xmin) / rng
                x01 = torch.clamp(x01, eps, 1 - eps)
                z = torch.log(x01) - torch.log1p(-x01)
                X[..., fi] = (z - st['mean']) / max(st['std'], eps)
            elif typ in ('minmax', 'bound01'):
                rng = max(st['max'] - st['min'], eps)
                y01 = (X[..., fi] - st['min']) / rng
                if typ == 'minmax':
                    X[..., fi] = y01 * 2.0 - 1.0
                else:  # bound01
                    X[..., fi] = y01 * 2.0 - 1.0
            elif typ == 'robust':
                X[..., fi] = (X[..., fi] - st['median']) / max(st['mad'], eps)
            else:
                # 디폴트 z
                X[..., fi] = (X[..., fi] - st['mean']) / max(st['std'], eps)
        return X

    @torch.no_grad()
    def inverse(self, Xn: torch.Tensor) -> torch.Tensor:
        Xn = Xn.clone()
        eps = 1e-6
        for fi, f in enumerate(self.feature_names):
            st = self.S[f]
            typ = st['type']
            if typ == 'z':
                Xn[..., fi] = Xn[..., fi] * st['std'] + st['mean']
            elif typ == 'log1p_z':
                z = Xn[..., fi] * st['std'] + st['mean']
                Xn[..., fi] = torch.expm1(z)
            elif typ == 'sqrt_z':
                z = Xn[..., fi] * st['std'] + st['mean']
                Xn[..., fi] = torch.square(torch.clamp(z, min=0.0))
            elif typ == 'logit_z':
                z = Xn[..., fi] * st['std'] + st['mean']
                x01 = torch.sigmoid(z)
                Xn[..., fi] = x01 * (st['xmax'] - st['xmin']) + st['xmin']
            elif typ in ('minmax', 'bound01'):
                y01 = (Xn[..., fi] + 1.0) / 2.0
                Xn[..., fi] = y01 * (st['max'] - st['min']) + st['min']
            elif typ == 'robust':
                Xn[..., fi] = Xn[..., fi] * st['mad'] + st['median']
            else:
                Xn[..., fi] = Xn[..., fi] * st['std'] + st['mean']
        return Xn


# =============================
# 유틸
# =============================
def month_to_sin_cos(month: int) -> Tuple[float, float]:
    angle = (float(month) - 1.0) * (2.0 * math.pi / 12.0)
    return math.sin(angle), math.cos(angle)

def build_model_input(xn: torch.Tensor, month: int, lat: float, lon: float) -> torch.Tensor:
    """
    xn: (L, F) normalized
    return: (1, L, F+4) with broadcasted [sin, cos, lat, lon]
    """
    L, F = xn.shape
    s, c = month_to_sin_cos(month)
    aux = torch.tensor([s, c, lat, lon], dtype=xn.dtype, device=xn.device)   # (4,)
    aux = aux.unsqueeze(0).repeat(L, 1)                                      # (L,4)
    xinp = torch.cat([xn, aux], dim=-1).unsqueeze(0)                         # (1,L,F+4)
    return xinp


# =============================
# 추론 서비스
# =============================
import json
import collections

class ClimatePredictor:
    def __init__(self, model_path, normalizer_stats_path, calibration_path, feature_names, sequence_length):
        self.feature_names = feature_names
        self.F = len(feature_names)
        self.L = sequence_length

        # --- Normalizer ---
        with open(normalizer_stats_path, "r") as f:
            stats = json.load(f)
        self.normalizer = Normalizer(stats, feature_names=feature_names)

        # --- Calibration ---
        with open(calibration_path, "r") as f:
            calib = json.load(f)
        self.dist = calib.get("dist", "gaussian")
        self.log_t_f = calib.get("log_t_f", None)      # list or None
        self.scalar_s = float(calib.get("scalar_s", 1.0))
        if self.log_t_f is not None:
            self.log_t_f = torch.tensor(self.log_t_f, dtype=torch.float32)

        # --- Model ---
        obj = torch.load(model_path, map_location="cpu")

        if isinstance(obj, collections.OrderedDict) or (isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys())):
            # state_dict만 저장된 케이스
            in_dim = self.F + 4
            self.model = FNN_ResLSTM_AttnHead(
                in_dim=in_dim, out_dim=self.F,
                pre_hidden=256, pre_out=128,
                lstm_hidden=256, num_layers=3,
                post_hidden=128, dropout=0.15, layer_norm=True
            )
            missing, unexpected = self.model.load_state_dict(obj, strict=False)
            if missing or unexpected:
                print(f"[warn] load_state_dict: missing={missing}, unexpected={unexpected}")
        else:
            # 전체 모델이 저장된 케이스
            self.model = obj

        self.device = DEVICE
        self.model.to(self.device).eval()

    # --- 내부: 보정 적용 ---
    def _apply_calibration(self, logvar: torch.Tensor) -> torch.Tensor:
        out = logvar
        if self.log_t_f is not None:
            out = out + 2.0 * self.log_t_f.to(logvar.device, dtype=logvar.dtype)
        if self.scalar_s is not None:
            out = out + 2.0 * math.log(float(self.scalar_s))
        return out

    # --- 1-스텝 예측 ---
    # --- 1-스텝 예측 (드롭인 교체용) ---
    def predict_next(
        self,
        x_window: np.ndarray,
        month_now: int,
        lat: float,
        lon: float,
        return_original_scale: bool = False,
        code: str = None,               # 선택: 지역 코드가 있으면 넘겨주세요
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        x_window: (L,F) 최근 L개의 '원스케일' 입력 (정규화는 내부에서 처리)
        month_now: 현재 월(1~12). 모델은 학습 시 '타깃 시점 보조피처'를 사용했으므로
                입력 보조피처는 (month_next)로 넣는 게 정합적입니다.
        반환: (mu, sigma, P68, P95, P997, month_next)
            - mu, sigma: 정규화 스케일
            - Pxx: 정규화 스케일에서의 신뢰구간 (F,2)
            - return_original_scale=True면 모두 원스케일로 변환해 반환
        """
        assert x_window.shape == (self.L, self.F), f"x_window must be ({self.L},{self.F})"

        # 0) 그룹 통계 선택 (있으면), 없으면 글로벌 사용
        try:
            self.normalizer.select_group(code=code, lat=float(lat), lon=float(lon))
        except Exception:
            # normalizer가 select_group을 지원하지 않는 구버전이어도 안전하게 진행
            pass

        # 1) 텐서화 + 정규화
        x_win = torch.tensor(x_window, dtype=torch.float32, device=self.device)  # (L,F)
        x_norm = self.normalizer.forward(x_win)                                  # (L,F)

        # 2) 보조피처: '다음 달'로 맞춰서 넣기(학습 시 타깃 시점 보조피처 사용)
        month_next = (int(month_now) % 12) + 1
        ang = (month_next - 1) * (2.0 * math.pi / 12.0)
        sinv = math.sin(ang)
        cosv = math.cos(ang)

        sin_m = torch.full((1, self.L, 1), sinv, dtype=torch.float32, device=self.device)
        cos_m = torch.full((1, self.L, 1), cosv, dtype=torch.float32, device=self.device)
        lat_t = torch.full((1, self.L, 1), float(lat), dtype=torch.float32, device=self.device)
        lon_t = torch.full((1, self.L, 1), float(lon), dtype=torch.float32, device=self.device)

        x_inp = torch.cat([x_norm.unsqueeze(0), sin_m, cos_m, lat_t, lon_t], dim=-1)  # (1,L,F+4)

        # 3) 추론
        self.model.eval()
        with torch.no_grad():
            mu, sec = self.model(x_inp)  # (1,F) each
        mu     = mu.squeeze(0)           # (F,)
        logvar = sec.squeeze(0)          # (F,)  (gaussian: logvar, laplace라면 logb)

        # 4) 캘리브레이션 적용 (per-feature log_t_f + scalar s)
        #    self._apply_calibration은 logvar를 받아 logvar'을 돌려주는 메서드라고 가정
        #    (per-feature가 없으면 그대로, scalar_s만 있으면 scalar만 적용)
        logvar = self._apply_calibration(logvar)  # (F,)

        # 5) 불확실성/구간 (Gaussian 가정)
        sigma = torch.clamp(torch.exp(0.5 * logvar), 1e-6, 1e6)      # (F,)
        P68   = torch.stack([mu - 1.0 * sigma, mu + 1.0 * sigma], dim=-1)  # (F,2)
        P95   = torch.stack([mu - 2.0 * sigma, mu + 2.0 * sigma], dim=-1)
        P997  = torch.stack([mu - 3.0 * sigma, mu + 3.0 * sigma], dim=-1)

        if not return_original_scale:
            # 정규화 스케일 그대로 반환
            return (mu.cpu().numpy(),
                    sigma.cpu().numpy(),
                    P68.cpu().numpy(),
                    P95.cpu().numpy(),
                    P997.cpu().numpy(),
                    month_next)

        # 6) 원스케일 역변환
        #    주의: 비선형 변환(log1p, sqrt, logit 등) 때문에 구간은 각 경계점을 따로 inverse 해야 함
        mu_o = self.normalizer.inverse(mu.unsqueeze(0)).squeeze(0)        # (F,)
        P68_lo  = self.normalizer.inverse(P68[..., 0].unsqueeze(0)).squeeze(0)
        P68_hi  = self.normalizer.inverse(P68[..., 1].unsqueeze(0)).squeeze(0)
        P95_lo  = self.normalizer.inverse(P95[..., 0].unsqueeze(0)).squeeze(0)
        P95_hi  = self.normalizer.inverse(P95[..., 1].unsqueeze(0)).squeeze(0)
        P997_lo = self.normalizer.inverse(P997[..., 0].unsqueeze(0)).squeeze(0)
        P997_hi = self.normalizer.inverse(P997[..., 1].unsqueeze(0)).squeeze(0)

        # 원스케일 sigma는 정확 정의가 애매(비선형)하므로, 구간 중심으로 근사 sigma 추출(옵션)
        # 여기선 단순히 (hi - lo)/2로 근사
        sigma_o = (P68_hi - P68_lo) * 0.5

        P68_o  = torch.stack([P68_lo,  P68_hi],  dim=-1)  # (F,2)
        P95_o  = torch.stack([P95_lo,  P95_hi],  dim=-1)
        P997_o = torch.stack([P997_lo, P997_hi], dim=-1)

        return (mu_o.cpu().numpy(),
                sigma_o.cpu().numpy(),
                P68_o.cpu().numpy(),
                P95_o.cpu().numpy(),
                P997_o.cpu().numpy(),
                month_next)
    # --- 멀티-스텝 롤아웃 (월만 앞으로 굴림, x는 관측창 유지 or 자가피드백 선택 가능) ---
    def rollout(
        self,
        x_window: np.ndarray,
        month_now: int,
        lat: float,
        lon: float,
        steps: int = 6,
        autoregressive: bool = False,
        return_original_scale: bool = False
    ):
        """
        steps 만큼 다음 달로 굴려가며 (mu, sigma, intervals) 수집
        autoregressive=False: 윈도우는 고정(관측창 유지, 계절 보조특성만 이동)
        autoregressive=True : 예측된 mu를 윈도우 뒤에 붙여 자가 피드백
        """
        L, F = self.L, self.F
        hist = np.array(x_window, dtype=np.float32).copy()
        out = {
            "mu": [], "sigma": [], "P68": [], "P95": [], "P997": [], "month": []
        }
        m = int(month_now)
        for _ in range(steps):
            mu, sigma, P68, P95, P997, m_next = self.predict_next(
                hist[-L:], m, lat, lon, return_original_scale=return_original_scale
            )
            out["mu"].append(mu); out["sigma"].append(sigma)
            out["P68"].append(P68); out["P95"].append(P95); out["P997"].append(P997)
            out["month"].append(m_next)
            m = m_next  # ← 반환된 월을 그대로 이어가기

            if autoregressive:
                if return_original_scale:
                    # 윈도우가 원스케일이면 그냥 mu(원스케일)를 붙이는 게 자연스럽습니다.
                    hist = np.vstack([hist, mu])
                else:
                    # 정규화 스케일을 사용 중이면 윈도우도 정규화 스케일이어야 일관됨
                    hist = np.vstack([hist, mu])
        # np.stack 변환
        out["mu"]   = np.stack(out["mu"], axis=0)    # (steps, F)
        out["sigma"]= np.stack(out["sigma"], axis=0) # (steps, F)
        out["P68"]  = np.stack(out["P68"], axis=0)   # (steps, F, 2)
        out["P95"]  = np.stack(out["P95"], axis=0)
        out["P997"] = np.stack(out["P997"], axis=0)
        out["month"]= np.array(out["month"], dtype=np.int32)
        return out