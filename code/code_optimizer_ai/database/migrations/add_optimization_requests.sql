CREATE TABLE IF NOT EXISTS optimization_requests (
    id UUID PRIMARY KEY,
    request_scope VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL,
    file_path TEXT,
    repository_url TEXT,
    representative_input_warning BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_optimization_requests_status
    ON optimization_requests(status);

CREATE INDEX IF NOT EXISTS idx_optimization_requests_created_at
    ON optimization_requests(created_at DESC);
