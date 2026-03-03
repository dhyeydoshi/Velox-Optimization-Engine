CREATE TABLE IF NOT EXISTS transition_memory (
    id UUID PRIMARY KEY,
    request_id UUID NOT NULL,
    hotspot_embedding vector(768) NOT NULL,
    family_tag VARCHAR(32) NOT NULL,
    original_code_hash VARCHAR(64) NOT NULL,
    transform_summary TEXT NOT NULL,
    code_diff TEXT NOT NULL,
    measured_runtime_delta DOUBLE PRECISION NOT NULL,
    measured_memory_delta DOUBLE PRECISION NULL,
    composite_score DOUBLE PRECISION NOT NULL,
    cv DOUBLE PRECISION NOT NULL,
    run_count INTEGER NOT NULL,
    model_id VARCHAR(64) NULL,
    prompt_template_version VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (run_count >= 0)
);

CREATE INDEX IF NOT EXISTS idx_transition_memory_family_score
    ON transition_memory(family_tag, composite_score DESC);

CREATE INDEX IF NOT EXISTS idx_transition_memory_created_at
    ON transition_memory(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_transition_memory_embedding
    ON transition_memory USING ivfflat (hotspot_embedding vector_cosine_ops)
    WITH (lists = 50);
