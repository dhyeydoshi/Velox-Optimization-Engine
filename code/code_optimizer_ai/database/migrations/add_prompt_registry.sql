CREATE TABLE IF NOT EXISTS prompt_template_registry (
    version VARCHAR(32) PRIMARY KEY,
    template_type VARCHAR(32) NOT NULL,
    template_text TEXT NOT NULL,
    avg_offspring_composite_score DOUBLE PRECISION NULL,
    avg_family_diversity_score DOUBLE PRECISION NULL,
    evaluation_sample_size INTEGER NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    notes TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_prompt_registry_type_active
    ON prompt_template_registry(template_type, is_active)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_prompt_registry_created_at
    ON prompt_template_registry(created_at DESC);
