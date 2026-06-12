-- Run in Supabase SQL Editor

-- ── Customers Table ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    phone       TEXT,
    address     JSONB,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ── Orders Table ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    id                 TEXT PRIMARY KEY,
    customer_id        TEXT NOT NULL REFERENCES customers(customer_id),
    status             TEXT NOT NULL,
    payment_status     TEXT NOT NULL,
    total              NUMERIC(10, 2) NOT NULL,
    subtotal           NUMERIC(10, 2),
    tax                NUMERIC(10, 2),
    shipping_cost      NUMERIC(10, 2),
    ordered_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    carrier            TEXT,
    estimated_delivery TIMESTAMP WITH TIME ZONE,
    delivered_at       TIMESTAMP WITH TIME ZONE,
    return_eligible    BOOLEAN DEFAULT FALSE,
    return_deadline    TIMESTAMP WITH TIME ZONE,
    notes              TEXT,
    items              JSONB,
    payment            JSONB,
    shipment           JSONB
);

-- ── Products Table ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS products (
    id               BIGINT PRIMARY KEY,
    title            TEXT NOT NULL,
    description      TEXT,
    price            NUMERIC(10, 2),
    image            TEXT NOT NULL,  -- Original image URL
    azure_image_url  TEXT            -- Final Azure Blob CDN URL
);

-- ── Customer Conversations (for Chat History UI) ─────────────────────────────
CREATE TABLE IF NOT EXISTS customer_conversations (
    conversation_id TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id),
    title           TEXT,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ── Checkpoint State Logs (for Readable JSON History) ─────────────────────────
CREATE TABLE IF NOT EXISTS checkpoint_state_logs (
    id                   BIGSERIAL PRIMARY KEY,
    conversation_id      TEXT NOT NULL,
    checkpoint_id        TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    step_node            TEXT,
    state_values         JSONB NOT NULL,
    metadata             JSONB,
    created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_state_logs_conversation ON checkpoint_state_logs(conversation_id);
