-- Run in Supabase SQL Editor
CREATE TABLE products (
    id               BIGINT PRIMARY KEY,
    title            TEXT NOT NULL,
    description      TEXT,
    price            NUMERIC(10, 2),
    image            TEXT NOT NULL,  -- Original image URL
    azure_image_url  TEXT            -- Final Azure Blob CDN URL
);

