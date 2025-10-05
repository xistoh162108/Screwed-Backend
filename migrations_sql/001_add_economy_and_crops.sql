-- Migration: Add session_economy and session_crop_stats tables
-- Created: 2025-10-05
-- Description: 세션별 경제 정보(잔액/통화)와 작물별 누적 통계(생산량/지속가능성 지표) 테이블 추가

-- ============================================
-- 1. Enum 타입 생성 (PostgreSQL)
-- ============================================

-- 작물 타입 Enum
CREATE TYPE crop_enum AS ENUM ('MAIZE', 'RICE', 'SOYBEAN', 'WHEAT');

-- ============================================
-- 2. session_economy 테이블 생성
-- ============================================

CREATE TABLE session_economy (
    id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL UNIQUE,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    balance NUMERIC(18, 2) NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Foreign Key
    CONSTRAINT fk_session_economy_session
        FOREIGN KEY (session_id)
        REFERENCES sessions(id)
        ON DELETE CASCADE
);

-- 인덱스
CREATE INDEX idx_session_economy_session_id ON session_economy(session_id);

-- 코멘트
COMMENT ON TABLE session_economy IS '세션별 경제 정보 (잔액, 통화)';
COMMENT ON COLUMN session_economy.id IS '경제 레코드 고유 ID (eco_{uuid})';
COMMENT ON COLUMN session_economy.session_id IS '세션 ID (외래키)';
COMMENT ON COLUMN session_economy.currency IS '통화 코드 (USD, KRW, EUR, JPY, CNY)';
COMMENT ON COLUMN session_economy.balance IS '현재 잔액';
COMMENT ON COLUMN session_economy.updated_at IS '마지막 업데이트 시각';

-- ============================================
-- 3. session_crop_stats 테이블 생성
-- ============================================

CREATE TABLE session_crop_stats (
    id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    crop crop_enum NOT NULL,

    -- 누적 생산량
    cumulative_production_tonnes NUMERIC(18, 3) NOT NULL DEFAULT 0,

    -- 지속가능성 지표 (누적)
    co2e_kg NUMERIC(18, 3) NOT NULL DEFAULT 0,      -- 탄소 배출량 (kg CO2e)
    water_m3 NUMERIC(18, 3) NOT NULL DEFAULT 0,     -- 물 사용량 (m³)
    fert_n_kg NUMERIC(18, 3) NOT NULL DEFAULT 0,    -- 질소 비료 (kg)
    fert_p_kg NUMERIC(18, 3) NOT NULL DEFAULT 0,    -- 인 비료 (kg)
    fert_k_kg NUMERIC(18, 3) NOT NULL DEFAULT 0,    -- 칼륨 비료 (kg)

    last_event_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Foreign Key
    CONSTRAINT fk_session_crop_stats_session
        FOREIGN KEY (session_id)
        REFERENCES sessions(id)
        ON DELETE CASCADE,

    -- Unique Constraint: 세션당 작물 종류별 1개 레코드만 허용
    CONSTRAINT uq_session_crop
        UNIQUE (session_id, crop)
);

-- 인덱스
CREATE INDEX idx_session_crop_stats_session_id ON session_crop_stats(session_id);
CREATE INDEX idx_session_crop_stats_crop ON session_crop_stats(crop);

-- 코멘트
COMMENT ON TABLE session_crop_stats IS '세션별 작물 통계 (누적 생산량, 지속가능성 지표)';
COMMENT ON COLUMN session_crop_stats.id IS '작물 통계 레코드 고유 ID (crop_{uuid})';
COMMENT ON COLUMN session_crop_stats.session_id IS '세션 ID (외래키)';
COMMENT ON COLUMN session_crop_stats.crop IS '작물 종류 (MAIZE, RICE, SOYBEAN, WHEAT)';
COMMENT ON COLUMN session_crop_stats.cumulative_production_tonnes IS '누적 생산량 (톤)';
COMMENT ON COLUMN session_crop_stats.co2e_kg IS '누적 탄소 배출량 (kg CO2 equivalent)';
COMMENT ON COLUMN session_crop_stats.water_m3 IS '누적 물 사용량 (m³)';
COMMENT ON COLUMN session_crop_stats.fert_n_kg IS '누적 질소(N) 비료 사용량 (kg)';
COMMENT ON COLUMN session_crop_stats.fert_p_kg IS '누적 인(P) 비료 사용량 (kg)';
COMMENT ON COLUMN session_crop_stats.fert_k_kg IS '누적 칼륨(K) 비료 사용량 (kg)';
COMMENT ON COLUMN session_crop_stats.last_event_at IS '마지막 이벤트 발생 시각';

-- ============================================
-- Rollback Script (필요시 사용)
-- ============================================

/*
-- 테이블 삭제 (역순)
DROP TABLE IF EXISTS session_crop_stats;
DROP TABLE IF EXISTS session_economy;

-- Enum 삭제
DROP TYPE IF EXISTS crop_enum;
*/
