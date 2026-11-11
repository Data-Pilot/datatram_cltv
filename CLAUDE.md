# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Five standalone Jupyter notebooks that compute DataTram's per-customer data-science outputs for every Shopify tenant and write them back to that tenant's ClickHouse database:

| Notebook | Output table | Cadence / run date |
|---|---|---|
| `01_ds_actuals` | `ds_actuals` (includes contact fields: email, first name, last name, phone) | monthly, `RUN_DATE` = 1st of current month |
| `02_ds_cltv` | `ds_cltv` (90-day churn probability + predicted LTV) | monthly |
| `03_ds_segmentation` | `ds_segmentation` (0/1 segment flags), then Datatram/MinIO data-science list refresh | monthly |
| `04_ds_predictions` | `ds_predictions` (LightGBM orders/spend forecasts, t1–t3) | monthly |
| `05_ds_recommendations` | `ds_recommendations` (item-item CF + content blend) | every two weeks, run date = today in each tenant's time zone; keeps the latest `KEEP_RUNS` (3) runs |

There is no shared pipeline module, build, or lint tooling. The numbering is just an order: each notebook reads the raw `shopify_*_transformed` tables itself, and none reads another notebook's output.

## Run metrics

Every notebook contains the complete inline metrics implementation in its first cell and creates its own `PipelineMetrics` instance there. A successful ClickHouse insert increments the run's row count, and a background sampler records the notebook process's peak resident memory. The notebook sends one best-effort OTLP/HTTP payload when processing finishes. The payload contains run status, duration, rows successfully written, peak memory in bytes, and (for successful runs only) the last-success timestamp. The `pipeline` attribute distinguishes `actuals`, `cltv`, `segmentation`, `predictions`, and `recommendations`. A metrics delivery or memory-sampling failure is logged but never fails a data pipeline. Configure `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`, `OTEL_EXPORTER_OTLP_METRICS_HEADERS`, `OTEL_SERVICE_NAME`, and `DEPLOY_ENV` in the job environment; see `.env.example` for non-secret examples. Because this implementation is intentionally duplicated, any metrics change must be applied to the first cell of all five scripts.

## Dev and prod

The same scripts run on both servers; every environment-specific value comes from that server's `.env` (template: `.env.example`). The scripts hold no dev or prod values at all:

- `DEPLOY_ENV`: environment label on the metrics (`dev` / `prod`).
- `PG_SCHEMA`: Postgres schema for the tenant query (`dev_datatram` / `prod_datatram`). Must be a plain identifier.
- `DATATRAM_BASE_URL`: 03's Datatram API URL (`https://dev.datatram.ai` / `https://app.datatram.ai`). If unset, 03 falls back to the `datatram-base-url` Vault secrets, then stops.

`DEPLOY_ENV` and `PG_SCHEMA` are read by `required_env()` in the Vault cell, so a script stops before touching any database if either is missing. Credentials and the Prometheus endpoint/auth also come from `.env`/Vault. Never hard-code an environment-specific value; add a new `.env` variable instead, in all five scripts.

## Running

```
pip install -r requirements.txt   # Python 3.10+
```

Credentials come from `.env` (`VAULT_ADDR`, `VAULT_ROLE_ID`, `VAULT_SECRET_ID`, optional `VAULT_KV_MOUNT`, default `datatram`). The notebooks were authored against a Jupyter kernel named `lahore`. For an ad-hoc or re-run date, uncomment the `RUN_DATE = datetime.strptime(...)` line in the run-date cell.

**Running a notebook writes to live tenant databases.** For each tenant, the main loop deletes any existing rows for the run date and inserts new ones. The final cell of every notebook only reads, printing row counts per `run_date` per tenant.

## Shared structure (duplicated in every notebook)

The cells follow the same order in every notebook: imports → Vault credentials → tenant list → run date/settings → `load_tenant_data` → model/feature logic → per-tenant write loop → verification. The Vault, tenant-list, and write-loop cells are copy-pasted into each notebook, so **a change to any of them has to be made in all five**.

- **Vault (OpenBao 2.1.1, hvac-compatible):** AppRole login once per run. Secret names use hyphens only (`secret_name()` maps `_` → `-`), and each value is stored as `{"value": ...}` (`VALUE_KEY`). The ClickHouse secrets are named `datatram-ssh-*` for historical reasons; they aren't SSH credentials.
- **Tenant list:** comes from Postgres (`tenants` + `tenant_properties`, Shopify source `source_id = 1`, status `connected`). The tenant's ClickHouse database name is `REPLACE(tenant_name, ' ', '_') || '_' || tenant_id`. The query's schema comes from `PG_SCHEMA` (see Dev and prod), not from the Vault `db_schema`. `tenant_properties` can hold duplicate rows per tenant (seen in prod), so 01–03 de-duplicate the tenant list (04/05 de-duplicate via the `{schema: tz}` dict). 04 and 05 also fetch each tenant's time zone (`tenant_properties.timezone_name`, falling back to `tenants.timezone`) and fail a tenant that has none.
- **Write loop:** `ALTER TABLE ... DELETE WHERE run_date = ...` with `settings={'mutations_sync': 1}` (so the insert can't race the mutation), then `insert_df`. Every tenant runs inside its own try/except and is added to `failed_tenants` on error; one bad tenant must not stop the rest. All five notebooks `CREATE TABLE IF NOT EXISTS` first, and fail a tenant early with a readable error if its source tables aren't loaded yet. 01–04 also backfill: they always rewrite `RUN_DATE`, and also write any of the `BACKFILL_MONTHS` (3) months before it that have no rows, so a new tenant starts with 4 months. 01 also migrates existing `ds_actuals` tables with `ADD COLUMN IF NOT EXISTS phone String AFTER email`; 03's Datatram list refresh expects that phone column when joining `ds_segmentation` to `ds_actuals`. Its `clean_contacts()` keeps a contact with a valid email OR a phone (at least 7 digits) and drops only those with neither, so a list can be smaller than its `ds_segmentation` segment (customers with no contact details) but phone-only customers are included. 05 doesn't: it runs every two weeks and keeps only 3 runs. When 05 produces no recommendations, a tenant with no previous runs is skipped (new, not enough history), and a tenant that already has runs fails (probably broken source data). Its previous runs are left in place either way. Each month runs on its own. A month without enough data is skipped and logged rather than failed. In 01 and 03 that means no customers. In 02 it's `data_shortfall()`: fewer than `MIN_TRAINING_CLASS` (3) buyers or churners in the training window. In 04 it's `training_shortfall()`: fewer than 2 training snapshots with customers, or a target with no positive labels. LightGBM's poisson/tweedie objectives raise an error on all-zero labels and on an empty train/validation set. The functions in these notebooks take `run_date` as a parameter; don't read the global `RUN_DATE` inside them. Existing tenants' tables differ slightly (e.g. `ds_segmentation` is `ReplacingMergeTree` on hinz and `MergeTree` on vera). The `*_DDL` constants copy the live column types exactly. 05 also migrates older tables that lack `run_date` and prunes old runs, but only after a successful insert.

## Data conventions and gotchas

- `closed_at IS NULL` selects the current version of each row. The `*_transformed` tables keep older versions too, so every query needs this filter.
- Order timestamps are compared with `< RUN_DATE`, never `<= CUT_OFF_DATE`: comparing a timestamp to a date means midnight, which would drop the last day.
- 01–03 use plain dates. 04 and 05 compare in tenant-local time: ClickHouse stores UTC, so boundaries go through `local_midnight()` in SQL and loaded timestamps go through `to_local()`.
- `clickhouse_connect` returns a frame with **no columns** when a query matches nothing. Use the `select_df()` helper, which rebuilds the expected columns. Before `insert_df`, cast pandas nullable extension dtypes (`Int64`, `string`) to plain numpy/object; see 05's `to_output()`.
- Existing table schemas constrain what you write. For example, `ds_segmentation` stores the id as `String` and `run_date` as `Date`, and those are sort-key columns, so their types can't be changed in place.
- "Spend" is defined differently on purpose in different notebooks: Shopify's lifetime `total_spent` (01), gross including voided/pending orders (02), `current_total_price` over fulfilled orders (03), and gross minus point-in-time refunds with cancelled orders excluded (04). Don't unify them.
- Models avoid look-ahead. Features use only data strictly before a snapshot or anchor date. In 04, refunds count only if made before `as_of` (`spend_as_of`). Thresholds (percentiles in 03, derived parameters in 05's `CONFIG`) are computed per tenant rather than fixed, so they hold for any store size or currency.
