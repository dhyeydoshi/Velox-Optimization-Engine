CREATE TABLE IF NOT EXISTS phase_a_candidate_evaluations (
    id UUID PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES optimization_requests(id) ON DELETE CASCADE,
    candidate_id UUID NOT NULL,
    generation_path VARCHAR(64) NOT NULL,
    family_tag VARCHAR(32) NOT NULL,
    gate_result VARCHAR(32) NOT NULL,
    gate_reason TEXT NOT NULL DEFAULT '',
    syntax_valid BOOLEAN NOT NULL,
    test_pass BOOLEAN NULL,
    run_count INTEGER NULL,
    warmup_runs_discarded INTEGER NULL,
    cv DOUBLE PRECISION NULL,
    runtime_delta_pct DOUBLE PRECISION NULL,
    memory_delta_pct DOUBLE PRECISION NULL,
    composite_score DOUBLE PRECISION NULL,
    representative_input_warning BOOLEAN NOT NULL DEFAULT FALSE,
    synthetic_fallback_penalty_applied BOOLEAN NOT NULL DEFAULT FALSE,
    benchmark_runs_min INTEGER NOT NULL DEFAULT 12,
    request_latency_ms INTEGER NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (request_id, candidate_id),
    CHECK (run_count IS NULL OR run_count >= 0),
    CHECK (warmup_runs_discarded IS NULL OR warmup_runs_discarded >= 0)
);

CREATE INDEX IF NOT EXISTS idx_phase_a_candidate_evals_request
    ON phase_a_candidate_evaluations(request_id);

CREATE INDEX IF NOT EXISTS idx_phase_a_candidate_evals_created_at
    ON phase_a_candidate_evaluations(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_phase_a_candidate_evals_gate_result
    ON phase_a_candidate_evaluations(gate_result);

CREATE INDEX IF NOT EXISTS idx_phase_a_candidate_evals_cv
    ON phase_a_candidate_evaluations(cv);
