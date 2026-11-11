#!/usr/bin/env python
# Converted from 05_ds_recommendations.ipynb. Run: python 05_ds_recommendations.py

# %%
## Import libraries

"""Best-effort OTLP/HTTP metrics for the CLTV batch pipelines."""

from __future__ import annotations

import atexit
import os
import threading
import time
from collections.abc import Mapping
from urllib.parse import unquote

import requests

try:
    import psutil
except ImportError:  # Keep observability optional if deployment missed the dependency.
    psutil = None


DEFAULT_SERVICE_NAME = "cltv-pipeline"
DEFAULT_ENVIRONMENT = "unknown"  # only used if DEPLOY_ENV is missing from .env
DEFAULT_TIMEOUT_SECONDS = 15
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.1


def _string_attribute(key: str, value: str) -> dict:
    return {"key": key, "value": {"stringValue": value}}


def _parse_otlp_headers(raw_headers: str) -> dict[str, str]:
    """Parse the OpenTelemetry comma-separated, percent-encoded header format."""
    headers: dict[str, str] = {}
    for item in raw_headers.split(","):
        key, separator, value = item.strip().partition("=")
        if not separator or not key:
            raise ValueError("invalid OTEL_EXPORTER_OTLP_METRICS_HEADERS format")
        headers[unquote(key)] = unquote(value)
    return headers


def push_metrics(
    values: Mapping[str, float | int],
    *,
    pipeline: str,
    units: Mapping[str, str] | None = None,
) -> bool:
    """Push one set of run-level metrics without ever failing the pipeline."""
    try:
        endpoint = os.environ["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"]
        headers = _parse_otlp_headers(
            os.environ["OTEL_EXPORTER_OTLP_METRICS_HEADERS"]
        )
        units = units or {}
        timestamp = str(time.time_ns())
        point_attributes = [_string_attribute("pipeline", pipeline)]

        payload = {
            "resourceMetrics": [
                {
                    "resource": {
                        "attributes": [
                            _string_attribute(
                                "service.name",
                                os.getenv("OTEL_SERVICE_NAME", DEFAULT_SERVICE_NAME),
                            ),
                            _string_attribute(
                                "deployment.environment",
                                os.getenv("DEPLOY_ENV", DEFAULT_ENVIRONMENT),
                            ),
                            _string_attribute("pipeline.name", pipeline),
                        ]
                    },
                    "scopeMetrics": [
                        {
                            "scope": {"name": "cltv"},
                            "metrics": [
                                {
                                    "name": name,
                                    "unit": units.get(name, ""),
                                    "gauge": {
                                        "dataPoints": [
                                            {
                                                "attributes": point_attributes,
                                                "asDouble": float(value),
                                                "timeUnixNano": timestamp,
                                            }
                                        ]
                                    },
                                }
                                for name, value in values.items()
                            ],
                        }
                    ],
                }
            ]
        }

        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        print(f"Metrics pushed for pipeline={pipeline}")
        return True
    except Exception as exc:
        # Never print headers or the payload: the authorization secret must not
        # appear in logs, and observability must never fail the data pipeline.
        print(f"WARNING: metrics push failed for pipeline={pipeline}: {exc}")
        return False


class _PeakMemorySampler:
    """Sample this notebook process's resident memory and retain the run peak."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._peak_bytes: int | None = None
        self._process = None
        self._thread = None

        if psutil is None:
            print("WARNING: peak memory sampling unavailable: psutil is not installed")
            return

        try:
            self._process = psutil.Process(os.getpid())
            self._sample()
            self._thread = threading.Thread(
                target=self._run,
                name="cltv-peak-memory-sampler",
                daemon=True,
            )
            self._thread.start()
        except Exception as exc:
            self._process = None
            print(f"WARNING: peak memory sampling unavailable: {exc}")

    def _sample(self) -> None:
        if self._process is None:
            return
        rss_bytes = int(self._process.memory_info().rss)
        if self._peak_bytes is None or rss_bytes > self._peak_bytes:
            self._peak_bytes = rss_bytes

    def _run(self) -> None:
        while not self._stop_event.wait(MEMORY_SAMPLE_INTERVAL_SECONDS):
            try:
                self._sample()
            except Exception:
                # A failed sample must not affect the pipeline. A final sample is
                # attempted by stop(), and the highest prior value is retained.
                pass

    def stop(self) -> int | None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        try:
            self._sample()
        except Exception:
            pass
        return self._peak_bytes


class PipelineMetrics:
    """Accumulate one pipeline run and push its metrics exactly once."""

    def __init__(self, pipeline: str) -> None:
        self.pipeline = pipeline
        self._started_at = time.perf_counter()
        self._memory_sampler = _PeakMemorySampler()
        self._rows_processed = 0
        self._finished = False
        self._lock = threading.Lock()
        atexit.register(self._finish_failed_at_exit)

    @property
    def rows_processed(self) -> int:
        return self._rows_processed

    def add_rows(self, rows: int) -> None:
        rows = int(rows)
        if rows < 0:
            raise ValueError("rows must not be negative")
        self._rows_processed += rows

    def finish(self, *, success: bool) -> bool:
        """Push status, duration, rows, and last-success time when successful."""
        with self._lock:
            if self._finished:
                return True
            self._finished = True

        try:
            atexit.unregister(self._finish_failed_at_exit)
        except Exception:
            pass

        peak_memory_bytes = self._memory_sampler.stop()
        values: dict[str, float | int] = {
            "cltv_pipeline_status": 1 if success else 0,
            "cltv_pipeline_duration": time.perf_counter() - self._started_at,
            "cltv_pipeline_rows_processed": self._rows_processed,
        }
        if peak_memory_bytes is not None:
            values["cltv_pipeline_peak_memory_bytes"] = peak_memory_bytes
        if success:
            values["cltv_pipeline_last_success_timestamp"] = time.time()

        return push_metrics(
            values,
            pipeline=self.pipeline,
            units={
                "cltv_pipeline_duration": "s",
                "cltv_pipeline_peak_memory_bytes": "By",
                "cltv_pipeline_last_success_timestamp": "s",
            },
        )

    def _finish_failed_at_exit(self) -> None:
        self.finish(success=False)

pipeline_metrics = PipelineMetrics("recommendations")

import os
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
import psycopg2
import clickhouse_connect
import hvac
from hvac.exceptions import InvalidPath
from dotenv import load_dotenv

pd.set_option('display.max_columns', None)

warnings.filterwarnings('ignore')

load_dotenv()

# %%
## Reading ClickHouse + Postgres credentials from Vault

VALUE_KEY = "value"

# Everything that differs between dev and prod comes from .env (see .env.example).
def required_env(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set in .env or the job environment")
    return value


DEPLOY_ENV = required_env("DEPLOY_ENV")  # environment label on the metrics, e.g. dev / prod
PG_SCHEMA = required_env("PG_SCHEMA")  # Postgres schema holding tenants, e.g. dev_datatram / prod_datatram
if not PG_SCHEMA.isidentifier():
    raise RuntimeError(f"PG_SCHEMA must be a plain schema name, got {PG_SCHEMA!r}")


def secret_name(name):
    """Normalise a config key to its Vault path (underscores -> hyphens)."""
    return name.replace('_', '-')


def get_secret(vault, mount, name):
    """Read a secret. Raises KeyError if it does not exist."""
    path = secret_name(name)
    try:
        resp = vault.secrets.kv.v2.read_secret_version(
            path=path,
            mount_point=mount,
            raise_on_deleted_version=True,
        )
    except InvalidPath:
        raise KeyError(f"secret not found: {path}")
    return resp['data']['data'][VALUE_KEY]


vault = hvac.Client(url=os.environ['VAULT_ADDR'])
vault.auth.approle.login(
    role_id=os.environ['VAULT_ROLE_ID'],
    secret_id=os.environ['VAULT_SECRET_ID'],
)
vault_mount = os.getenv('VAULT_KV_MOUNT', 'datatram').strip('/')

# ClickHouse credentials (secret names kept their original "ssh-" prefix in Vault)
CH_HOST = get_secret(vault, vault_mount, 'datatram-ssh-host')
CH_USERNAME = get_secret(vault, vault_mount, 'datatram-ssh-username')
CH_PASSWORD = get_secret(vault, vault_mount, 'datatram-ssh-password')
CH_PORT = 8123  # default HTTP port

# Postgres credentials
db_name = get_secret(vault, vault_mount, 'datatram-postgresql-db-name')
db_host = get_secret(vault, vault_mount, 'datatram-postgresql-host')
db_password = get_secret(vault, vault_mount, 'datatram-postgresql-password')
db_port = get_secret(vault, vault_mount, 'datatram-postgresql-port')
db_schema = get_secret(vault, vault_mount, 'datatram-postgresql-schema')
db_user = get_secret(vault, vault_mount, 'datatram-postgresql-user')


def get_clickhouse_client(database):
    """ClickHouse client for a given tenant database, using the credentials above."""
    return clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USERNAME,
        password=CH_PASSWORD,
        database=database,
    )

# %%
## List of tenants with Shopify connection status 'connected', and their time zones

# --- Establish connection ---
conn = psycopg2.connect(
    host=db_host,
    port=int(db_port),
    dbname=db_name,
    user=db_user,
    password=db_password,
    options=f'-c search_path={db_schema}'
)

# --- Query ---
query = f'''
SELECT
    t.tenant_id,
    t.tenant_name,
    REPLACE(t.tenant_name, ' ', '_') || '_' || t.tenant_id AS schema_name,
    t.created_at,
    -- Shopify source's own time zone first, then the tenant's
    COALESCE(NULLIF(tp.timezone_name, ''), NULLIF(t.timezone, '')) AS timezone,
    tp.source_connection_status AS shopify_connection_status
FROM "{PG_SCHEMA}".tenants t
LEFT JOIN "{PG_SCHEMA}".tenant_properties tp
    ON tp.tenant_id = t.tenant_id
   AND tp.source_id = (
        SELECT source_id
        FROM "{PG_SCHEMA}".data_sources
        WHERE source_id = 1
   );
'''

tenants = pd.read_sql_query(query, conn)

conn.close()

tenants = tenants[tenants['shopify_connection_status'] == 'connected'].copy()
tenants = dict(zip(tenants['schema_name'], tenants['timezone']))  # {schema_name: time zone}

print(f"Found {len(tenants)} tenants with Shopify connection status 'connected'.")
print(tenants)

# %%
## Run date and settings

# Runs every two weeks. RUN_DATE = None uses today's date in each tenant's own
# time zone; set a date to re-run a specific day (that run is replaced).
RUN_DATE = None
# RUN_DATE = datetime.strptime('2026-09-15', '%Y-%m-%d').date()  ## for ad-hoc runs

TABLE = 'ds_recommendations'
KEEP_RUNS = 3  # runs (distinct run_dates) kept per tenant; older ones are dropped

# Rows written before the run_date column existed get this value when it is added
LEGACY_RUN_DATE = '1970-01-01'

# Used only for tenants that don't have the table yet. Tables created before
# run_date existed keep their original sort key (customer_id, recommendation_rank).
RECOMMENDATIONS_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE}
(
    customer_id          UInt64,
    product_id           UInt64,
    product_name         String,
    recommendation_rank  UInt8,
    recommendation_score Float32,
    generated_at         DateTime,
    run_date             Date
)
ENGINE = MergeTree
ORDER BY (run_date, customer_id, recommendation_rank)
"""

# %%
## Reading a tenant's customers, orders, line items and products

def select_df(client, columns, from_where):
    """SELECT `columns` as a DataFrame. clickhouse_connect returns a frame with
    no columns at all when nothing matches (e.g. a month with no abandoned
    checkouts), so an empty result is rebuilt with the expected columns."""
    df = client.query_df(f"SELECT {', '.join(columns)} {from_where}")
    names = [c.split(' AS ')[-1].strip() for c in columns]  # "expr AS name" -> name
    return df if len(df.columns) else pd.DataFrame(columns=names)


def local_midnight(day, tz):
    """SQL for the instant of local midnight at the start of `day` in `tz`.
    ClickHouse stores every timestamp in UTC, so day/month boundaries have to
    be converted to the tenant's time zone to fall on local midnight."""
    return f"toDateTime('{day} 00:00:00', '{tz}')"


def to_local(ts, tz):
    """UTC timestamps -> the tenant's local wall-clock time (tz-naive), so all
    date arithmetic downstream happens in local days."""
    ts = pd.to_datetime(ts)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize('UTC')
    return ts.dt.tz_convert(tz).dt.tz_localize(None)


def local_today(tz):
    """Today's date in the tenant's time zone."""
    return pd.Timestamp.now(tz=tz).date()


def table_columns(client, table):
    """Column names of `table` in the tenant's database."""
    cols = client.query_df(f"""
        SELECT name FROM system.columns
        WHERE database = currentDatabase() AND table = '{table}'
    """)
    return set(cols['name']) if len(cols.columns) else set()


def pick_columns(client, table, required, optional):
    """Required columns plus whichever optional ones this tenant has. A missing
    required column fails the tenant; missing optional ones are returned so the
    filters that depend on them can be reported as skipped."""
    have = table_columns(client, table)
    missing_required = [c for c in required if c not in have]
    if missing_required:
        raise ValueError(f"{table} is missing required columns: {missing_required}")
    return (required + [c for c in optional if c in have],
            [f"{table}.{c}" for c in optional if c not in have])


# Required columns, and optional ones that only switch on an extra filter / feature
CUSTOMER_COLS = (['customer_system_shopify_id'], ['created_at_date'])
ORDER_COLS = (['order_system_shopify_id', 'customer_system_shopify_id', 'order_created_at'],
              ['order_cancelled_at', 'order_financial_status', 'order_source_name'])
LINE_ITEM_COLS = (['order_system_shopify_id', 'product_system_shopify_id', 'line_item_quantity'],
                  ['gift_card', 'product_exists'])
PRODUCT_COLS = (['product_system_shopify_id', 'product_title', 'product_status'],
                ['product_type', 'vendor_name', 'product_tags', 'product_sku'])


def load_tenant_data(client, tz, run_date):
    """Customers created before `run_date`, orders placed before local midnight,
    their line items, and the current product catalog. Guest checkouts (customer
    id 0) are excluded because they are not one customer."""
    end = local_midnight(run_date, tz)
    cut_off = pd.Timestamp(run_date).date() - pd.Timedelta(days=1)
    customers_in_window = f"""closed_at IS NULL
             AND customer_system_shopify_id != 0"""
    orders_in_window = f"""closed_at IS NULL
             AND customer_system_shopify_id != 0
             AND order_created_at < {end}"""

    customer_cols, missing_c = pick_columns(client, 'shopify_customers_transformed', *CUSTOMER_COLS)
    customers = select_df(client, customer_cols, f"""FROM shopify_customers_transformed
           WHERE {customers_in_window}""")
    if 'created_at_date' in customers.columns:
        customers = customers[pd.to_datetime(customers['created_at_date']).dt.date <= cut_off]

    order_cols, missing_o = pick_columns(client, 'shopify_orders_transformed', *ORDER_COLS)
    orders = select_df(client, order_cols, f"""FROM shopify_orders_transformed
           WHERE {orders_in_window}""")

    line_cols, missing_l = pick_columns(client, 'shopify_line_items_transformed', *LINE_ITEM_COLS)
    line_items = select_df(client, line_cols, f"""FROM shopify_line_items_transformed
           WHERE closed_at IS NULL
             AND order_system_shopify_id IN (
                 SELECT order_system_shopify_id FROM shopify_orders_transformed
                 WHERE {orders_in_window})""")

    product_cols, missing_p = pick_columns(client, 'shopify_products_transformed', *PRODUCT_COLS)
    products = select_df(client, product_cols, """FROM shopify_products_transformed
           WHERE closed_at IS NULL""")

    orders['order_created_at'] = to_local(orders['order_created_at'], tz)
    return customers, orders, line_items, products, missing_c + missing_o + missing_l + missing_p

# %%
## Recommender

"""
DataTram Shopify Recommender -- Generation
==========================================

Tenant-agnostic. Runs on any Shopify retail business sharing this schema.
No compiled third-party dependencies -- numpy / pandas / scipy only.

Model: category-aware item-item cosine similarity, blended from two channels
    CF      -- co-occurrence in the customer x product matrix
    CONTENT -- shared product_type / vendor_name / product_tags (tf-idf cosine)
Serving first diversifies by product category, then fills sparse customers
from tenant-level popular products so every known customer can receive rows.
The blend weight is DERIVED from how much co-purchase signal the tenant
actually has, so a business with deep multi-item baskets leans on CF and a
young business with mostly single-item orders leans on content instead.

Every threshold the original Hinz build hardcoded is now derived at fit time:
bulk-buyer cut, recency half-life, quantity cap, neighbour count, top_n,
candidate depth, draft-order handling, and variant grouping.

generate_recommendations(customers, orders, line_items, products, run_date) returns
(recs, diag), where recs is a DataFrame:
    customer_id | product_id | product_name | product_category
                | recommendation_rank | recommendation_score | generated_at
and diag is a dict of what happened (filter counts, derived parameters,
coverage, and any filter skipped because its column is missing).

Customers without personal signal fall back to diversified popular products.
Nothing is written to disk. Nothing is printed.
"""

# =========================================================================
# CONFIG -- policy only. Anything scale-dependent is DERIVED below.
# =========================================================================
CONFIG = {
    # ---- order validity ----
    "exclude_cancelled": True,
    "exclude_financial_status": ["voided", "refunded", "partially_refunded"],
    # Draft orders are noise at some tenants and the primary sales channel at
    # others (made-to-order businesses book most sales manually). "auto" drops
    # them only when they are a marginal share of orders.
    "exclude_draft_orders": "auto",              # True | False | "auto"
    "draft_auto_max_share": 0.05,
    "draft_source_names": ["shopify_draft_order"],

    # ---- line item validity ----
    "require_product_exists": True,
    "invalid_product_id_sentinels": [0],
    "exclude_gift_cards": True,
    "min_quantity": 1,

    # ---- product eligibility ----
    "require_nonblank_title": True,
    "serve_active_only": True,
    "active_status_values": ["active"],

    # ---- interaction weighting ----
    "recency_floor": 0.3,
    "cap_orders_per_pair": 3,
    "quantity_cap_quantile": 0.99,
    "use_weighted_matrix": False,   # binary co-occurrence is the sturdier signal

    # ---- similarity blend ----
    "use_content_channel": True,
    "content_fields": ["product_type", "vendor_name"],
    "content_list_fields": ["product_tags"],
    "content_list_sep": ",",
    "min_cf_weight": 0.15,
    "max_cf_weight": 0.85,

    # ---- serving ----
    "min_distinct_products_serve": 1,
    "exclude_previously_purchased": True,
    "top_n_cap": 10,
    "top_n_max_share_of_catalog": 0.25,
    "candidate_depth_multiple": 6,

    # ---- category diversification ----
    "category_field": "product_type",
    "uncategorized_label": "__uncategorized__",
    "max_per_category": 2,
    "category_candidate_multiplier": 4,
    "cold_start_score_factor": 0.35,

    # ---- variant grouping (auto-discovered, no word lists) ----
    "variant_discovery": True,
    "variant_min_collisions": 2,
    "variant_min_token_support": 2,
    "variant_collapse_min_ratio": 1.5,
    "max_per_base_product": 2,
    "sku_min_coverage": 0.5,
    "base_key_max_tokens": 6,

    # ---- compute ----
    "similarity_chunk": 512,
    "score_batch": 2048,
}

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


# =========================================================================
# HELPERS
# =========================================================================
def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _lower(s):
    return s.astype(str).str.strip().str.lower()


def _s(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def _has(df, col):
    return col in df.columns


def _tokenize(text):
    return [t for t in _TOKEN_RE.split(_s(text).lower()) if t]


def _skip(diag, name):
    """Note a configured filter that couldn't run because its column is missing."""
    diag["filters_skipped"].append(name)


# =========================================================================
# 1. INTERACTIONS
# =========================================================================
def build_interaction_lines(orders, line_items, products, cfg, diag):
    o = orders.copy()
    diag["orders_in_window"] = len(o)
    diag["filters_skipped"] = []

    if cfg["exclude_cancelled"]:
        if _has(o, "order_cancelled_at"):
            o = o[o["order_cancelled_at"].isna()]
        else:
            _skip(diag, "exclude_cancelled")

    if cfg["exclude_financial_status"]:
        if _has(o, "order_financial_status"):
            bad = [x.lower() for x in cfg["exclude_financial_status"]]
            o = o[~_lower(o["order_financial_status"]).isin(bad)]
        else:
            _skip(diag, "exclude_financial_status")

    drop_drafts = cfg["exclude_draft_orders"]
    if drop_drafts and not _has(o, "order_source_name"):
        _skip(diag, "exclude_draft_orders")
    if _has(o, "order_source_name"):
        is_draft = o["order_source_name"].astype(str).isin(cfg["draft_source_names"])
        share = float(is_draft.mean()) if len(o) else 0.0
        if drop_drafts == "auto":
            drop_drafts = share <= cfg["draft_auto_max_share"]
        diag["draft_order_share"] = round(share, 4)
        diag["draft_orders_excluded"] = bool(drop_drafts)
        if drop_drafts:
            o = o[~is_draft]

    o = o[o["customer_system_shopify_id"].notna()]
    diag["orders_valid"] = len(o)

    keep = ["order_system_shopify_id", "customer_system_shopify_id", "order_created_at"]
    li = line_items.merge(o[keep], on="order_system_shopify_id", how="inner")

    if cfg["exclude_gift_cards"]:
        if _has(li, "gift_card"):
            li = li[~li["gift_card"].astype(bool)]
        else:
            _skip(diag, "exclude_gift_cards")
    if cfg["require_product_exists"]:
        if _has(li, "product_exists"):
            li = li[li["product_exists"].astype(bool)]
        else:
            _skip(diag, "require_product_exists")

    li = li[li["product_system_shopify_id"].notna()]
    if cfg["invalid_product_id_sentinels"]:
        li = li[~li["product_system_shopify_id"].isin(cfg["invalid_product_id_sentinels"])]
    li = li[_num(li["line_item_quantity"]) >= cfg["min_quantity"]]

    pm = products[products["product_system_shopify_id"].notna()].drop_duplicates(
        "product_system_shopify_id")
    if cfg["require_nonblank_title"]:
        pm = pm[pm["product_title"].fillna("").astype(str).str.strip() != ""]
    li = li[li["product_system_shopify_id"].isin(set(pm["product_system_shopify_id"]))]

    li = li.rename(columns={
        "customer_system_shopify_id": "customer_id",
        "product_system_shopify_id": "product_id",
        "order_system_shopify_id": "order_id",
        "order_created_at": "order_ts",
    })
    li["order_ts"] = pd.to_datetime(li["order_ts"], errors="coerce")
    li["qty_raw"] = _num(li["line_item_quantity"]).fillna(1)

    lines = li[["customer_id", "product_id", "order_id", "order_ts", "qty_raw"]]
    lines = lines.dropna(subset=["order_ts"]).reset_index(drop=True)
    diag["interaction_lines"] = len(lines)
    return lines, pm


def derive_params(lines, prod_meta, cfg, diag):
    """Recompute, from this tenant, every constant the first build hardcoded."""
    ppc = lines.groupby("customer_id")["product_id"].nunique()
    span_days = max((lines["order_ts"].max() - lines["order_ts"].min()).days, 1)
    n_items = int(lines["product_id"].nunique())

    n_servable = int(
        _lower(prod_meta["product_status"]).isin(cfg["active_status_values"]).sum()
    ) if cfg["serve_active_only"] else len(prod_meta)
    n_servable = max(n_servable, 1)

    basket = lines.groupby("order_id")["product_id"].nunique()
    multi_item_share = float((basket >= 2).mean()) if len(basket) else 0.0

    p = {
        # 3x the 99th percentile reproduces the hand-tuned cut on a deep-basket
        # tenant and goes correctly inert where the deepest basket is 4 items
        "bulk_threshold": int(max(np.ceil(3 * ppc.quantile(0.99)), 10)),
        # decay tied to observed history rather than a fixed horizon
        "recency_half_life_days": int(np.clip(round(span_days / 3), 30, 365)),
        "quantity_cap": int(max(_num(lines["qty_raw"]).quantile(
            cfg["quantity_cap_quantile"]), 1)),
        # neighbour count is meaningless above catalog size
        "topk_neighbours": int(np.clip(n_items // 5, 10, 200)),
        "top_n": int(np.clip(
            min(cfg["top_n_cap"],
                int(np.floor(n_servable * cfg["top_n_max_share_of_catalog"]))),
            1, cfg["top_n_cap"])),
        # how far to trust co-occurrence vs content, from observed basket depth
        "cf_weight": float(np.clip(multi_item_share,
                                   cfg["min_cf_weight"], cfg["max_cf_weight"])),
        "multi_item_order_share": round(multi_item_share, 4),
        "history_days": span_days,
        "n_servable_products": n_servable,
    }
    p["candidate_depth"] = p["top_n"] * cfg["candidate_depth_multiple"]
    diag["derived_params"] = dict(p)
    return p


def aggregate_pairs(lines, cfg, params, ref):
    """Per (customer, product) weights; recency is measured back from `ref`."""
    g = lines.groupby(["customer_id", "product_id"], sort=False).agg(
        n_orders=("order_id", "nunique"),
        total_qty=("qty_raw", "sum"),
        last_ts=("order_ts", "max"),
    ).reset_index()

    g["total_qty"] = g["total_qty"].clip(upper=params["quantity_cap"])
    g["n_orders_capped"] = g["n_orders"].clip(upper=cfg["cap_orders_per_pair"])

    days = (ref - g["last_ts"]).dt.total_seconds().div(86400).clip(lower=0)
    decay = np.exp(-np.log(2) * days / params["recency_half_life_days"])
    recency = cfg["recency_floor"] + (1 - cfg["recency_floor"]) * decay
    freq = 1.0 + np.log1p((g["n_orders_capped"] - 1 + g["total_qty"] - 1).clip(lower=0))
    g["weight"] = (freq * recency).astype(np.float32)
    return g


# =========================================================================
# 2. SIMILARITY (chunked, pruned, sparse -- never materialises n x n dense)
# =========================================================================
def cosine_topk(M, topk, chunk):
    """Row-wise cosine similarity of sparse M, keeping topk per row.

    Diagonal is zeroed. Returned as CSR so a 20k-product catalog costs
    topk * n_items entries rather than n_items squared.
    """
    M = sp.csr_matrix(M, dtype=np.float32)
    norms = np.sqrt(M.multiply(M).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    Mn = sp.diags(1.0 / norms) @ M

    n = Mn.shape[0]
    k = min(topk, max(n - 1, 1))
    rows, cols, vals = [], [], []

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = np.asarray((Mn[start:stop] @ Mn.T).todense(), dtype=np.float32)
        for i in range(block.shape[0]):
            r = block[i]
            r[start + i] = 0.0
            if k < n:
                cut = np.argpartition(-r, kth=k - 1)[:k]
            else:
                cut = np.arange(n)
            keep = cut[r[cut] > 0]
            if keep.size:
                rows.append(np.full(keep.size, start + i, dtype=np.int32))
                cols.append(keep.astype(np.int32))
                vals.append(r[keep])

    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n), dtype=np.float32)


def build_content_matrix(prod_meta, item_ids, cfg, diag):
    """tf-idf weighted item x feature matrix from categorical metadata."""
    pm = prod_meta.set_index("product_system_shopify_id").reindex(item_ids)
    docs = []
    for _, row in pm.iterrows():
        feats = []
        for col in cfg["content_fields"]:
            v = _s(row.get(col)).lower()
            if v:
                feats.append(f"{col}={v}")
        for col in cfg["content_list_fields"]:
            for part in _s(row.get(col)).lower().split(cfg["content_list_sep"]):
                part = part.strip()
                if part:
                    feats.append(f"{col}={part}")
        docs.append(feats)

    vocab = {}
    for d in docs:
        for f in d:
            vocab.setdefault(f, len(vocab))
    diag["content_vocab"] = len(vocab)
    diag["items_with_content"] = int(sum(1 for d in docs if d))
    if not vocab:
        return None

    rows, cols = [], []
    for i, d in enumerate(docs):
        for f in set(d):
            rows.append(i)
            cols.append(vocab[f])
    F = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(item_ids), len(vocab)))

    df = np.asarray(F.sum(axis=0)).ravel()
    idf = np.log((1 + len(item_ids)) / (1 + df)) + 1.0
    return F @ sp.diags(idf.astype(np.float32))


# =========================================================================
# 3. VARIANT GROUPING (discovered from the catalog, no hardcoded word lists)
# =========================================================================
def discover_variant_tokens(titles, cfg):
    """Find tokens that merely distinguish variants of one underlying product.

    A token is a variant token if deleting it makes two or more otherwise
    distinct titles collide. On an apparel catalog this surfaces colours and
    sizes; on a catalog where those words carry real meaning it surfaces
    nothing. Either way the answer comes from the tenant's own titles.
    """
    toks = [_tokenize(t) for t in titles]
    groups = defaultdict(list)
    for i, tl in enumerate(toks):
        if len(tl) < 2:
            continue
        for k, tok in enumerate(tl):
            key = tuple(tl[:k]) + tuple(tl[k + 1:])
            groups[key].append((i, tok))

    counts = Counter()
    for members in groups.values():
        if len({i for i, _ in members}) >= cfg["variant_min_collisions"]:
            for _, tok in members:
                counts[tok] += 1

    return {t for t, c in counts.items() if c >= cfg["variant_min_token_support"]}, toks


def build_base_keys(prod_meta, item_ids, cfg, diag):
    pm = prod_meta.set_index("product_system_shopify_id").reindex(item_ids)
    titles = pm["product_title"].tolist()

    # SKU prefixes group variants where the tenant maintains structured SKUs
    sku_keys = None
    if _has(prod_meta, "product_sku"):
        skus = [_s(v) for v in pm["product_sku"]]
        coverage = sum(1 for s in skus if s) / max(len(skus), 1)
        prefixes = ["".join(ch for ch in s if ch.isalnum()).upper()[:8] for s in skus]
        distinct = len({p for p in prefixes if len(p) >= 4})
        diag["sku_coverage"] = round(coverage, 4)
        if coverage >= cfg["sku_min_coverage"] and distinct > 1:
            sku_keys = ["sku:" + p if len(p) >= 4 else None for p in prefixes]

    variant_tokens, toks = (set(), [_tokenize(t) for t in titles])
    if cfg["variant_discovery"]:
        variant_tokens, toks = discover_variant_tokens(titles, cfg)
    diag["variant_tokens_discovered"] = len(variant_tokens)

    keys = []
    for i, tl in enumerate(toks):
        if sku_keys is not None and sku_keys[i]:
            keys.append(sku_keys[i])
            continue
        stripped = [t for t in tl if t not in variant_tokens]
        if not stripped:
            stripped = tl
        keys.append("ttl:" + " ".join(stripped[:cfg["base_key_max_tokens"]]))

    keys = np.array(keys, dtype=object)
    ratio = len(item_ids) / max(len(set(keys)), 1)
    enabled = ratio >= cfg["variant_collapse_min_ratio"]
    diag["variants_per_base"] = round(ratio, 3)
    diag["variant_collapse_enabled"] = bool(enabled)
    return keys, enabled


# =========================================================================
# 4. SERVING HELPERS
# =========================================================================
def customer_ids_from(customers):
    """Known customers to serve. Falls back to an empty array when the customer
    frame is unavailable so the historical item-item path remains testable."""
    if customers is None or not len(customers) or "customer_system_shopify_id" not in customers.columns:
        return np.array([], dtype=np.int64)
    ids = _num(customers["customer_system_shopify_id"]).dropna().astype("int64")
    ids = ids[ids != 0].drop_duplicates()
    return np.array(sorted(ids), dtype=np.int64)


def category_values(prod_meta, item_ids, cfg):
    """Category aligned to the item index. Shopify product_type is the default
    because it is the nearest existing category field in this schema."""
    field = cfg["category_field"]
    label = cfg["uncategorized_label"]
    pm = prod_meta.set_index("product_system_shopify_id").reindex(item_ids)
    if field not in pm.columns:
        return np.array([label] * len(item_ids), dtype=object)
    vals = pm[field].fillna("").astype(str).str.strip()
    vals = vals.mask(vals == "", label)
    return vals.to_numpy(dtype=object)


def popular_product_order(pairs, cand_mask):
    """Tenant-level product popularity used for cold-start and sparse customers."""
    pop = pairs.groupby("product_id")["weight"].sum().sort_values(ascending=False)
    return pop


def append_ranked_products(cust, candidate_idx, score_lookup, item_ids, title_arr,
                           category_arr, base_keys, max_per_base, cfg, top_n,
                           seen_products, out, score_factor=1.0):
    """Append a category-diverse top-N list for one customer.

    The first pass caps categories, which prevents ten products from one
    category when alternatives exist. The second pass relaxes only the category
    cap so tiny catalogs can still fill available slots.
    """
    kept = 0
    used_products = set()
    per_base = {}
    per_category = {}

    def try_add(j, enforce_category):
        nonlocal kept
        product_id = item_ids[j]
        if product_id in used_products or product_id in seen_products:
            return False
        score = score_lookup(j)
        if not np.isfinite(score) or score <= 0:
            return False
        if max_per_base is not None:
            bk = base_keys[j]
            if per_base.get(bk, 0) >= max_per_base:
                return False
        cat = category_arr[j]
        if enforce_category and per_category.get(cat, 0) >= cfg["max_per_category"]:
            return False

        used_products.add(product_id)
        if max_per_base is not None:
            per_base[bk] = per_base.get(bk, 0) + 1
        per_category[cat] = per_category.get(cat, 0) + 1
        kept += 1
        out["customer_id"].append(cust)
        out["product_id"].append(product_id)
        out["product_name"].append(title_arr[j])
        out["product_category"].append(cat)
        out["recommendation_rank"].append(kept)
        out["recommendation_score"].append(float(score) * score_factor)
        return True

    for enforce_category in (True, False):
        for j in candidate_idx:
            if kept >= top_n:
                return kept
            try_add(j, enforce_category)
    return kept


# =========================================================================
# 5. MAIN
# =========================================================================
def generate_recommendations(customers, orders, line_items, products, run_date, cfg=CONFIG):
    """Recommendations as of `run_date` (recency reference), plus diagnostics."""
    diag = {}
    generated_at = pd.Timestamp.utcnow().tz_localize(None).floor("s")
    all_customer_ids = customer_ids_from(customers)
    diag["all_customers"] = len(all_customer_ids)

    empty = pd.DataFrame({
        "customer_id": pd.Series(dtype="int64"),
        "product_id": pd.Series(dtype="int64"),
        "product_name": pd.Series(dtype="string"),
        "product_category": pd.Series(dtype="string"),
        "recommendation_rank": pd.Series(dtype="int16"),
        "recommendation_score": pd.Series(dtype="float32"),
        "generated_at": pd.Series(dtype="datetime64[ns]"),
    })

    def bail(status):
        diag["status"] = status
        return empty, diag

    lines, prod_meta = build_interaction_lines(orders, line_items, products, cfg, diag)
    if not len(lines):
        return bail("no_interactions")

    params = derive_params(lines, prod_meta, cfg, diag)
    seen_by_customer = {
        int(c): set(g["product_id"].astype("int64"))
        for c, g in lines.groupby("customer_id", sort=False)
    }

    # wholesale / staff accounts distort item-item structure, but those customers
    # can still receive tenant-popular fallback recommendations below
    ppc = lines.groupby("customer_id")["product_id"].nunique()
    bulk = set(ppc[ppc > params["bulk_threshold"]].index)
    lines = lines[~lines["customer_id"].isin(bulk)]
    diag["bulk_buyers_excluded"] = len(bulk)
    if not len(lines):
        return bail("no_interactions_after_bulk_filter")

    pairs = aggregate_pairs(lines, cfg, params, pd.Timestamp(run_date))

    user_ids = np.array(sorted(pairs["customer_id"].unique()))
    item_ids = np.array(sorted(pairs["product_id"].unique()))
    u2i = {u: i for i, u in enumerate(user_ids)}
    p2i = {p: j for j, p in enumerate(item_ids)}
    diag["customers_in_matrix"] = len(user_ids)
    diag["products_in_matrix"] = len(item_ids)

    X = sp.csr_matrix(
        (pairs["weight"].to_numpy(np.float32),
         (pairs["customer_id"].map(u2i).to_numpy(), pairs["product_id"].map(p2i).to_numpy())),
        shape=(len(user_ids), len(item_ids)), dtype=np.float32)
    X_fit = X if cfg["use_weighted_matrix"] else (X > 0).astype(np.float32)

    # ---- blended item-item similarity ----
    S_cf = cosine_topk(X_fit.T, params["topk_neighbours"], cfg["similarity_chunk"])
    alpha = params["cf_weight"]

    S_content = None
    if cfg["use_content_channel"]:
        F = build_content_matrix(prod_meta, item_ids, cfg, diag)
        if F is not None and F.nnz:
            S_content = cosine_topk(F, params["topk_neighbours"], cfg["similarity_chunk"])

    if S_content is None:
        alpha = 1.0
        S = S_cf
    else:
        S = (alpha * S_cf + (1.0 - alpha) * S_content).tocsr()
    diag["cf_weight_applied"] = round(float(alpha), 4)
    diag["similarity_nnz"] = int(S.nnz)

    # ---- candidate pool ----
    status = prod_meta.set_index("product_system_shopify_id")["product_status"].reindex(item_ids)
    if cfg["serve_active_only"]:
        cand_mask = _lower(status.fillna("")).isin(cfg["active_status_values"]).to_numpy()
    else:
        cand_mask = np.ones(len(item_ids), dtype=bool)
    diag["servable_products"] = int(cand_mask.sum())
    if not cand_mask.any():
        return bail("no_servable_products")

    base_keys, collapse_on = build_base_keys(prod_meta, item_ids, cfg, diag)
    max_per_base = cfg["max_per_base_product"] if collapse_on else None
    category_arr = category_values(prod_meta, item_ids, cfg)
    diag["servable_categories"] = int(len(set(category_arr[cand_mask])))

    # titles aligned to the item index, so lookup during ranking is positional
    title_arr = (prod_meta.set_index("product_system_shopify_id")["product_title"]
                 .reindex(item_ids).fillna("").astype(str).to_numpy())

    # ---- who is scoreable ----
    dpc = pairs.groupby("customer_id")["product_id"].nunique()
    eligible = dpc[dpc >= cfg["min_distinct_products_serve"]].index
    rows_eligible = np.array(sorted(u2i[u] for u in eligible if u in u2i))
    diag["customers_eligible"] = len(rows_eligible)

    top_n = params["top_n"]
    depth = min(
        max(params["candidate_depth"],
            top_n * cfg["candidate_depth_multiple"] * cfg["category_candidate_multiplier"]),
        len(item_ids),
    )
    blocked = ~cand_mask
    batch = cfg["score_batch"]

    out = {
        "customer_id": [],
        "product_id": [],
        "product_name": [],
        "product_category": [],
        "recommendation_rank": [],
        "recommendation_score": [],
    }

    served_customers = set()
    if len(rows_eligible):
        for start in range(0, len(rows_eligible), batch):
            rws = rows_eligible[start:start + batch]
            Sc = np.asarray((X_fit[rws] @ S).todense(), dtype=np.float32)
            Sc[:, blocked] = -np.inf
            if cfg["exclude_previously_purchased"]:
                Sc[X_fit[rws].nonzero()] = -np.inf

            k = min(depth, Sc.shape[1])
            part = np.argpartition(-Sc, kth=k - 1, axis=1)[:, :k]
            ridx = np.arange(Sc.shape[0])[:, None]
            order = np.argsort(-Sc[ridx, part], axis=1)
            top = part[ridx, order]

            for i, r in enumerate(rws):
                cust = int(user_ids[r])
                seen = seen_by_customer.get(cust, set()) if cfg["exclude_previously_purchased"] else set()
                kept = append_ranked_products(
                    cust, top[i], lambda j, row=i: Sc[row, j], item_ids, title_arr,
                    category_arr, base_keys, max_per_base, cfg, top_n, seen, out)
                if kept:
                    served_customers.add(cust)

    # ---- cold-start / sparse fallback for every known customer ----
    all_known = all_customer_ids if len(all_customer_ids) else user_ids.astype(np.int64)
    pop = popular_product_order(pairs, cand_mask)
    pop_idx = np.array([p2i[p] for p in pop.index if p in p2i and cand_mask[p2i[p]]],
                       dtype=np.int64)
    pop_scores = {p2i[p]: float(s) for p, s in pop.items() if p in p2i}

    cold_start = 0
    for cust in all_known:
        cust = int(cust)
        if cust in served_customers:
            continue
        seen = seen_by_customer.get(cust, set()) if cfg["exclude_previously_purchased"] else set()
        kept = append_ranked_products(
            cust, pop_idx, lambda j: pop_scores.get(j, 0.0), item_ids, title_arr,
            category_arr, base_keys, max_per_base, cfg, top_n, seen, out,
            score_factor=cfg["cold_start_score_factor"])
        if kept:
            cold_start += 1
            served_customers.add(cust)

    recs = pd.DataFrame({
        "customer_id": pd.array(out["customer_id"], dtype="int64"),
        "product_id": pd.array(out["product_id"], dtype="int64"),
        "product_name": pd.array(out["product_name"], dtype="string"),
        "product_category": pd.array(out["product_category"], dtype="string"),
        "recommendation_rank": pd.array(out["recommendation_rank"], dtype="int16"),
        "recommendation_score": np.array(out["recommendation_score"], dtype=np.float32),
        "generated_at": generated_at,
    })

    diag["cold_start_customers"] = cold_start
    diag["customers_served"] = int(recs["customer_id"].nunique()) if len(recs) else 0
    diag["recommendations"] = len(recs)
    diag["distinct_products_recommended"] = (
        int(recs["product_id"].nunique()) if len(recs) else 0)
    diag["catalog_coverage"] = round(
        diag["distinct_products_recommended"] / max(diag["servable_products"], 1), 4)
    diag["served_share_of_known"] = round(
        diag["customers_served"] / max(len(all_known), 1), 4)
    diag["avg_categories_per_customer"] = (
        round(float(recs.groupby("customer_id")["product_category"].nunique().mean()), 3)
        if len(recs) else 0.0)
    diag["status"] = "success" if len(recs) else "no_output"
    return recs, diag

# %%
## Writing and reporting results

def prepare_table(client):
    """Make sure the table exists and has run_date, and remove rows written
    before run_date existed (they carry LEGACY_RUN_DATE). The clean-up only
    finds rows once per tenant; after that it is a no-op."""
    client.command(RECOMMENDATIONS_DDL)
    client.command(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS run_date Date")
    legacy = int(client.command(
        f"SELECT count() FROM {TABLE} WHERE run_date = toDate('{LEGACY_RUN_DATE}')"))
    if legacy:
        client.command(f"ALTER TABLE {TABLE} DELETE WHERE run_date = toDate('{LEGACY_RUN_DATE}')",
                       settings={'mutations_sync': 1})
        print(f"  Removed {legacy:,} legacy rows written before run_date existed")


def prune_old_runs(client, table=TABLE, keep=KEEP_RUNS):
    """Keep only the `keep` most recent run_dates and drop everything older.
    Called after a successful insert, so a failed run never costs an old one;
    re-running an existing date replaces it and doesn't count as a new run."""
    runs = client.query_df(f"SELECT DISTINCT run_date FROM {table} ORDER BY run_date DESC")
    if len(runs) <= keep:
        return
    run_dates = [pd.Timestamp(d).date() for d in runs['run_date']]
    oldest_kept, dropped = run_dates[keep - 1], run_dates[keep:]
    client.command(f"ALTER TABLE {table} DELETE WHERE run_date < toDate('{oldest_kept}')",
                   settings={'mutations_sync': 1})
    print(f"  Dropped {len(dropped)} old run(s) {[str(d) for d in dropped]}; "
          f"keeping the latest {keep}")


def to_output(recs, run_date):
    """Upload frame. Extension dtypes are cast to plain numpy/object first --
    clickhouse_connect does not handle pandas' nullable Int64/string types."""
    return pd.DataFrame({
        "customer_id": recs["customer_id"].astype("int64").to_numpy(),
        "product_id": recs["product_id"].astype("int64").to_numpy(),
        "product_name": recs["product_name"].fillna("").astype(str).to_numpy(),
        "recommendation_rank": recs["recommendation_rank"].astype("uint8").to_numpy(),
        "recommendation_score": recs["recommendation_score"].astype("float32").to_numpy(),
        "generated_at": pd.to_datetime(recs["generated_at"]).dt.floor("s"),
        "run_date": pd.Timestamp(run_date).date(),
    })


DIAG_SUMMARY_KEYS = [
    'status', 'orders_valid', 'interaction_lines', 'draft_orders_excluded',
    'bulk_buyers_excluded', 'all_customers', 'customers_eligible', 'customers_served',
    'cold_start_customers', 'recommendations', 'distinct_products_recommended',
    'catalog_coverage', 'served_share_of_known', 'avg_categories_per_customer',
    'servable_categories', 'cf_weight_applied', 'variant_collapse_enabled',
]


def print_diagnostics(diag):
    """One-screen summary of a tenant's run."""
    print("  " + ", ".join(f"{k}={diag[k]}" for k in DIAG_SUMMARY_KEYS if k in diag))
    derived = diag.get('derived_params', {})
    if derived:
        print("  derived: " + ", ".join(f"{k}={v}" for k, v in derived.items()))
    if diag.get('filters_skipped'):
        print(f"  WARNING: filters skipped (column missing): {diag['filters_skipped']}")

# %%
## Generating and writing recommendations for every connected tenant

SOURCE_TABLES = ('shopify_customers_transformed', 'shopify_orders_transformed',
                 'shopify_line_items_transformed', 'shopify_products_transformed')

failed_tenants = []
skipped_tenants = []  # no previous runs and no recommendations yet (not enough history)

for tenant_db, tz in tenants.items():
    print(f"Processing tenant: {tenant_db}")
    client = None

    try:
        if not tz:
            raise ValueError("no time zone set in tenant_properties.timezone_name or tenants.timezone")
        run_date = RUN_DATE or local_today(tz)
        client = get_clickhouse_client(tenant_db)

        missing = [t for t in SOURCE_TABLES if not int(client.command(f'EXISTS TABLE {t}'))]
        if missing:
            raise ValueError(f"{missing} do not exist yet (source data not loaded)")

        # Preparing the table first, so every tenant has one even before its
        # first successful run
        prepare_table(client)
        previous_runs = int(client.command(f"SELECT count(DISTINCT run_date) FROM {TABLE}"))

        customers, orders, line_items, products, missing = load_tenant_data(client, tz, run_date)
        print(f"  run date {run_date} ({tz}): {len(customers)} customers, "
              f"{len(orders)} orders, {len(line_items)} line items, {len(products)} products")
        if missing:
            print(f"  Optional columns missing: {missing}")

        recs, diag = generate_recommendations(customers, orders, line_items, products, run_date)
        print_diagnostics(diag)
        if not len(recs):
            # A new tenant without enough history yet is expected; a tenant that
            # had recommendations before and now has none points to a data problem
            if not previous_runs:
                print(f"  SKIPPED: no recommendations yet (status: {diag['status']}); "
                      "a new tenant needs more order history")
                skipped_tenants.append(tenant_db)
                continue
            raise ValueError(f"no recommendations generated (status: {diag['status']}); "
                             "previous runs left in place")
        output = to_output(recs, run_date)

        # Removing existing records for this run date, if any.
        # mutations_sync=1 waits for the DELETE to finish before returning, so
        # the insert below can't race an in-flight mutation.
        try:
            client.command(
                f"""
                ALTER TABLE {TABLE}
                DELETE WHERE run_date = toDate('{run_date}')
                """,
                settings={'mutations_sync': 1},
            )
        except Exception:
            print(f"  FAILED to clear existing rows for {tenant_db}; leaving old data in place, skipping insert.")
            raise

        # Appending recommendations to the database
        try:
            client.insert_df(TABLE, output)
            pipeline_metrics.add_rows(len(output))
        except Exception:
            print(f"  FAILED to insert new rows for {tenant_db}; {TABLE} has NO rows for {run_date} until this tenant is re-run.")
            raise

        # Keeping only the latest KEEP_RUNS runs. The new run is already written,
        # so a failure here only means an extra old run stays until next time.
        try:
            prune_old_runs(client)
        except Exception as exc:
            print(f"  WARNING: could not drop old runs for {tenant_db}: {exc}")

        print(f"  OK: {tenant_db}")

    except Exception as exc:
        print(f"  ERROR processing {tenant_db}: {exc}")
        failed_tenants.append(tenant_db)

    finally:
        if client is not None:
            client.close()

print("\nDone.")
if failed_tenants:
    print(f"Failed tenants ({len(failed_tenants)}): {failed_tenants}")
else:
    print("All tenants processed successfully.")
if skipped_tenants:
    print(f"Skipped tenants, no recommendations yet ({len(skipped_tenants)}): {skipped_tenants}")

pipeline_metrics.finish(success=not failed_tenants)

# %%
## Verifying the upload, per tenant

for tenant_db in tenants:
    client = get_clickhouse_client(tenant_db)
    try:
        print(f"Tenant: {tenant_db}")
        if not int(client.command(f"EXISTS TABLE {TABLE}")):
            print(f"  no {TABLE} table yet\n")
            continue
        print(client.query_df(f"""
        SELECT
            run_date,
            COUNT(*) AS records,
            uniqExact(customer_id) AS customers,
            max(generated_at) AS generated_at
        FROM {TABLE}
        GROUP BY run_date
        ORDER BY run_date ASC
        """))
        duplicates = client.command(f"""
        SELECT count() FROM (
            SELECT customer_id, recommendation_rank
            FROM {TABLE}
            WHERE run_date = (SELECT max(run_date) FROM {TABLE})
            GROUP BY customer_id, recommendation_rank
            HAVING count() > 1
        )
        """)
        print(f"Duplicate (customer, rank) rows in the latest run: {duplicates}")
        print()
    finally:
        client.close()
