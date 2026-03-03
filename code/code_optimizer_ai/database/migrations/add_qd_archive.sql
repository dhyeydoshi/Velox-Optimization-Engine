CREATE TABLE IF NOT EXISTS qd_archive (
    id UUID PRIMARY KEY,
    family_tag VARCHAR(32) NOT NULL,
    performance_tier VARCHAR(16) NOT NULL,
    best_candidate_id UUID NULL,
    best_request_id UUID NULL,
    composite_score DOUBLE PRECISION NOT NULL,
    measured_runtime_delta DOUBLE PRECISION NOT NULL,
    code_pattern_hash VARCHAR(64) NULL,
    transform_summary TEXT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (family_tag, performance_tier)
);

CREATE INDEX IF NOT EXISTS idx_qd_archive_family_tier
    ON qd_archive(family_tag, performance_tier);
