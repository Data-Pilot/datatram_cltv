#!/usr/bin/env python
# Converted from 04_ds_predictions.ipynb. Run: python 04_ds_predictions.py

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

pipeline_metrics = PipelineMetrics("predictions")

import os
import warnings
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import psycopg2
import clickhouse_connect
import hvac
import lightgbm as lgb
from dateutil.relativedelta import relativedelta
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
## Set the run date to the first day of the current month

RUN_DATE = date.today().replace(day=1)
# RUN_DATE = datetime.strptime('2026-06-01', '%Y-%m-%d').date()  ## for ad-hoc runs

TABLE = 'ds_predictions'

# RUN_DATE is always (re)written. Any of the BACKFILL_MONTHS months before it
# that has no rows yet is written too, so a newly connected tenant starts with
# RUN_DATE plus 3 months of history, and a month that failed is retried next run.
BACKFILL_MONTHS = 3

# Forecast settings
ORDERS_START_DATE = '2025-01-01'            # earliest order used (tenant-local date)
MIN_HISTORY_DAYS = 90                       # a training snapshot needs this much order history before it
HORIZONS = [(0, 30), (30, 60), (60, 90)]    # t1, t2, t3: [lo, hi) days after the snapshot
LABEL_DAYS = HORIZONS[-1][1]                # a training snapshot needs this much observed future
FEATURE_LOOKBACKS = [30, 60, 90, 180, 365]  # rolling feature windows (days)
T0_DAYS = 30                                # "actual" window: the T0_DAYS before the run date

# LightGBM: orders -> Poisson (counts), spend -> Tweedie (zero-inflated amounts)
LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  verbose=-1, n_jobs=-1)
TWEEDIE_POWER = 1.3
NUM_BOOST_ROUND = 600
EARLY_STOPPING_ROUNDS = 50

OUTPUT_COLS = ['customer_system_shopify_id',
               'pred_total_spent_t1', 'pred_total_spent_t2', 'pred_total_spent_t3',
               'pred_orders_t1', 'pred_orders_t2', 'pred_orders_t3',
               'actual_total_spent_t0', 'actual_orders_t0', 'run_date']

# Created in tenants that don't have the table yet (same schema as the existing ones)
PREDICTIONS_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE}
(
    customer_system_shopify_id UInt64,
    pred_total_spent_t1 Float64,
    pred_total_spent_t2 Float64,
    pred_total_spent_t3 Float64,
    pred_orders_t1 UInt32,
    pred_orders_t2 UInt32,
    pred_orders_t3 UInt32,
    actual_total_spent_t0 Float64,
    actual_orders_t0 UInt32,
    run_date Date
)
ENGINE = ReplacingMergeTree
ORDER BY (run_date, customer_system_shopify_id)
"""

# %%
## Reading a tenant's customers, orders, sales and refunds

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


def load_tenant_data(client, tz, run_date):
    """Customers created before `run_date`, their orders from ORDERS_START_DATE up
    to (not including) `run_date` with each order's gross spend, and the refunds
    on those orders made before `run_date` - all boundaries at local midnight.
    Cancelled orders are excluded (they are not purchases)."""
    start, end = local_midnight(ORDERS_START_DATE, tz), local_midnight(run_date, tz)
    in_window = f"order_created_at >= {start} AND order_created_at < {end}"

    customers = select_df(
        client,
        ['customer_system_shopify_id', 'created_at'],
        f"""FROM shopify_customers_transformed
           WHERE closed_at IS NULL AND created_at < {end}""",
    )

    orders = select_df(
        client,
        ['customer_system_shopify_id', 'order_system_shopify_id', 'order_status', 'order_created_at'],
        f"""FROM shopify_orders_transformed
           WHERE closed_at IS NULL AND {in_window}""",
    )
    orders = orders[~orders['order_status'].eq('Cancelled').fillna(False)]

    # Order value before refunds; refunds are applied point-in-time (see spend_as_of).
    # The sales table has one row per line item per refund, so a line refunded
    # twice appears twice: collapse to one row per line first, or its discount
    # would be subtracted once per refund.
    gross = select_df(
        client,
        ['order_system_shopify_id',
         """any(order_gross) - sum(line_discount) + any(fees) + any(duties)
            + any(shipping) + any(tax) + any(adj) AS gross_spend"""],
        f"""FROM (
               SELECT order_system_shopify_id, line_item_system_shopify_id,
                      any(line_item_discount) AS line_discount,
                      any(order_total_gross_amount) AS order_gross,
                      any(additional_fees_amount) AS fees,
                      any(duties_amount) AS duties,
                      any(shipping_discounted_price_amount) AS shipping,
                      any(order_tax) AS tax,
                      any(adjustment) AS adj
               FROM shopify_sales_analytics_reporting
               WHERE {in_window}
               GROUP BY order_system_shopify_id, line_item_system_shopify_id
           )
           GROUP BY order_system_shopify_id""",
    )

    refunds = select_df(
        client,
        ['order_system_shopify_id', 'refund_created_at',
         'sum(refund_total_amount) AS refund_amount'],
        f"""FROM shopify_sales_analytics_reporting
           WHERE {in_window}
             AND refund_total_amount != 0
             AND refund_created_at < {end}
           GROUP BY order_system_shopify_id, refund_created_at""",
    )

    customers['created_at'] = to_local(customers['created_at'], tz)
    orders['order_created_at'] = to_local(orders['order_created_at'], tz)
    refunds['refund_created_at'] = to_local(refunds['refund_created_at'], tz)

    orders = orders.merge(gross, on='order_system_shopify_id', how='left')
    missing = int(orders['gross_spend'].isna().sum())
    if missing:
        print(f"  WARNING: {missing} orders have no sales row; their spend is counted as 0")
    orders['gross_spend'] = orders['gross_spend'].fillna(0.0).astype(float)

    return customers, orders.reset_index(drop=True), refunds

# %%
## Point-in-time spend

def spend_as_of(orders, refunds, as_of, report=False):
    """Each order's spend as it was known at `as_of`: gross spend minus the
    refunds made before `as_of`. A refund made later is not yet known then, so
    features never see refunds from their own future. Aligned with `orders`.

    Spend is floored at 0: a refund larger than the order's recorded value is a
    data inconsistency, and negative targets would break the Tweedie models.
    With report=True, orders more than a cent below 0 are reported."""
    known = refunds[refunds['refund_created_at'] < pd.Timestamp(as_of)]
    refunded = known.groupby('order_system_shopify_id')['refund_amount'].sum()
    spend = orders['gross_spend'] - orders['order_system_shopify_id'].map(refunded).fillna(0.0)
    negative = spend < -0.01  # ignore float noise on fully refunded orders
    if report and negative.any():
        print(f"  WARNING: {int(negative.sum())} orders have refunds above their value "
              f"(total {spend[negative].sum():,.2f}); their spend is floored at 0")
    return spend.clip(lower=0)

# %%
## Features, labels and the training set

def build_features(customers, orders, refunds, snapshot):
    """One feature row per customer who existed before `snapshot`, using only
    orders placed - and refunds made - strictly before it."""
    snap = pd.Timestamp(snapshot)
    existing = customers[customers['created_at'] < snap]
    hist = orders[orders['order_created_at'] < snap].copy()
    hist['spend'] = spend_as_of(hist, refunds, snap)

    base = pd.DataFrame({
        'customer_system_shopify_id': existing['customer_system_shopify_id'].values,
        'tenure_days': (snap - existing['created_at']).dt.days.values,
    })

    by_customer = hist.groupby('customer_system_shopify_id')
    lifetime = by_customer.agg(
        n_orders_all=('order_system_shopify_id', 'size'),
        spend_all=('spend', 'sum'),
        spend_mean=('spend', 'mean'),
        spend_max=('spend', 'max'),
        first_order=('order_created_at', 'min'),
        last_order=('order_created_at', 'max'),
    )
    lifetime['recency_days'] = (snap - lifetime['last_order']).dt.days
    lifetime['lifespan_days'] = (lifetime['last_order'] - lifetime['first_order']).dt.days
    base = base.merge(lifetime.drop(columns=['first_order', 'last_order']),
                      left_on='customer_system_shopify_id', right_index=True, how='left')

    # Rolling-window counts & spend
    for lb in FEATURE_LOOKBACKS:
        window = hist[hist['order_created_at'] >= snap - pd.Timedelta(days=lb)]
        rolled = window.groupby('customer_system_shopify_id').agg(
            **{f'n_orders_{lb}d': ('order_system_shopify_id', 'size'),
               f'spend_{lb}d': ('spend', 'sum')})
        base = base.merge(rolled, left_on='customer_system_shopify_id', right_index=True, how='left')

    # Seasonality of the period being forecast (the snapshot is its first day)
    base['snap_month'] = snap.month
    base['snap_quarter'] = snap.quarter

    # Counts/spend -> 0; never ordered -> large recency sentinel
    count_spend_cols = [c for c in base.columns if c.startswith(('n_orders', 'spend'))]
    base[count_spend_cols] = base[count_spend_cols].fillna(0)
    base['recency_days'] = base['recency_days'].fillna(9999)
    base['lifespan_days'] = base['lifespan_days'].fillna(0)
    return base


def build_labels(orders, snapshot):
    """Orders and net spend per customer in each horizon [snap + lo, snap + hi)."""
    snap = pd.Timestamp(snapshot)
    labels = []
    for i, (lo, hi) in enumerate(HORIZONS, start=1):
        window = orders[(orders['order_created_at'] >= snap + pd.Timedelta(days=lo)) &
                        (orders['order_created_at'] < snap + pd.Timedelta(days=hi))]
        labels.append(window.groupby('customer_system_shopify_id').agg(
            **{f'y_orders_t{i}': ('order_system_shopify_id', 'size'),
               f'y_spend_t{i}': ('net_spend', 'sum')}))
    return pd.concat(labels, axis=1)


def training_snapshots(run_date):
    """Month starts with at least MIN_HISTORY_DAYS of orders before them and a
    full LABEL_DAYS of observed future before `run_date`."""
    first = pd.Timestamp(ORDERS_START_DATE) + pd.Timedelta(days=MIN_HISTORY_DAYS)
    last = pd.Timestamp(run_date) - pd.Timedelta(days=LABEL_DAYS)
    snapshots = []
    snap = last.replace(day=1)
    while snap >= first:
        snapshots.append(snap)
        snap -= relativedelta(months=1)
    return sorted(snapshots)


def build_training_set(customers, orders, refunds, run_date):
    """Rolling-origin training rows: features at each snapshot + the labels
    that followed it. Only customers who existed at a snapshot are included, so
    snapshots before the tenant's first customer add no rows (checked by
    training_shortfall)."""
    snapshots = training_snapshots(run_date)
    frames = []
    for snap in snapshots:
        labels = build_labels(orders, snap)
        frame = build_features(customers, orders, refunds, snap).merge(
            labels, left_on='customer_system_shopify_id', right_index=True, how='left')
        frame[labels.columns] = frame[labels.columns].fillna(0)
        frame['snapshot'] = snap
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

# %%
## Models, predictions and actuals

def feature_columns(frame):
    return [c for c in frame.columns
            if c not in ('customer_system_shopify_id', 'snapshot') and not c.startswith('y_')]


def train_models(train):
    """6 LightGBM models - orders (Poisson) and spend (Tweedie) for each horizon -
    trained on every snapshot but the latest, which is used for early stopping."""
    cols = feature_columns(train)
    val_snap = train['snapshot'].max()
    tr, va = train[train['snapshot'] < val_snap], train[train['snapshot'] == val_snap]

    models = {}
    for i in range(1, len(HORIZONS) + 1):
        for target, objective, alias in [(f'y_orders_t{i}', 'poisson', f'orders_t{i}'),
                                         (f'y_spend_t{i}', 'tweedie', f'spend_t{i}')]:
            params = dict(LGB_PARAMS, objective=objective)
            if objective == 'tweedie':
                params['tweedie_variance_power'] = TWEEDIE_POWER
            dtr = lgb.Dataset(tr[cols], tr[target])
            dva = lgb.Dataset(va[cols], va[target], reference=dtr)
            model = lgb.train(params, dtr, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
            models[alias] = model
            print(f"  trained {alias:10s} best_iter={model.best_iteration:<4d} "
                  f"val_score={list(model.best_score['valid_0'].values())[0]:.4f}")
    return models


def training_shortfall(train):
    """Why train_models can't fit on `train`, or None if it can. It needs at least
    two snapshots with customers (all but the latest to train on, the latest for
    early stopping), and every target needs a positive label in the training
    snapshots - LightGBM's poisson/tweedie objectives reject labels summing to 0."""
    snapshots = sorted(train['snapshot'].unique()) if len(train) else []
    if len(snapshots) < 2:
        return (f"only {len(snapshots)} training snapshot(s) with customers "
                "(need 2: one to train on, one for early stopping)")
    tr = train[train['snapshot'] < snapshots[-1]]
    targets = [c for c in train.columns if c.startswith('y_')]
    empty = [t for t in targets if not tr[t].sum() > 0]
    if empty:
        return (f"no positive labels for {empty} in training snapshots "
                f"before {pd.Timestamp(snapshots[-1]).date()}")
    return None


def largest_remainder_round(x):
    """Round a float array to ints whose sum equals round(sum(x)), allocating the
    rounding surplus to the largest fractional parts (Hamilton method), so the
    total of predicted orders matches the sum of the raw expectations."""
    x = np.asarray(x, dtype=float)
    floor = np.floor(x)
    remainder = x - floor
    deficit = int(np.round(x.sum())) - int(floor.sum())  # how many +1s to hand out
    out = floor.copy()
    if deficit > 0:
        out[np.argsort(-remainder)[:deficit]] += 1  # largest fractions first
    return out.astype(int)


def predict(models, features):
    """Predicted orders (integers) and spend per customer for each horizon."""
    X = features[feature_columns(features)]
    result = features[['customer_system_shopify_id']].copy()
    for i in range(1, len(HORIZONS) + 1):
        raw_orders = np.clip(models[f'orders_t{i}'].predict(X), 0, None)
        result[f'pred_orders_t{i}'] = largest_remainder_round(raw_orders)
        result[f'pred_total_spent_t{i}'] = np.clip(models[f'spend_t{i}'].predict(X), 0, None)
        print(f"  t{i}: predicted orders {raw_orders.sum():,.1f} "
              f"(integer {result[f'pred_orders_t{i}'].sum():,}), "
              f"spend {result[f'pred_total_spent_t{i}'].sum():,.0f}")
    return result


def actual_t0(orders, run_date):
    """Actual orders and net spend per customer in the T0_DAYS before `run_date`."""
    end = pd.Timestamp(run_date)
    window = orders[(orders['order_created_at'] >= end - pd.Timedelta(days=T0_DAYS)) &
                    (orders['order_created_at'] < end)]
    return window.groupby('customer_system_shopify_id').agg(
        actual_total_spent_t0=('net_spend', 'sum'),
        actual_orders_t0=('order_system_shopify_id', 'size'))

# %%
## Forecasting and writing predictions for every connected tenant

SOURCE_TABLES = ('shopify_customers_transformed', 'shopify_orders_transformed',
                 'shopify_sales_analytics_reporting')


def write_predictions(client, run_date, predictions):
    """Replace the rows for `run_date` with `predictions`."""
    # mutations_sync=1 waits for the DELETE to finish before returning, so
    # the insert below can't race an in-flight mutation.
    try:
        client.command(
            f"""
            ALTER TABLE {TABLE}
            DELETE WHERE toDate(run_date) = toDate('{run_date}')
            """,
            settings={'mutations_sync': 1},
        )
    except Exception:
        print(f"    FAILED to clear existing rows for {run_date}; leaving old data in place, skipping insert.")
        raise

    try:
        client.insert_df(TABLE, predictions)
        pipeline_metrics.add_rows(len(predictions))
    except Exception:
        print(f"    FAILED to insert new rows; {TABLE} has NO rows for {run_date} until it is re-run.")
        raise


def months_to_write(client):
    """The BACKFILL_MONTHS months before RUN_DATE that have no rows yet, oldest
    first, followed by RUN_DATE itself (always rewritten)."""
    earlier = [RUN_DATE - relativedelta(months=k) for k in range(BACKFILL_MONTHS, 0, -1)]
    existing = client.query_df(f"""
        SELECT DISTINCT toDate(run_date) AS run_day FROM {TABLE}
        WHERE run_date >= toDate('{RUN_DATE - relativedelta(months=BACKFILL_MONTHS)}')
    """)
    have = {pd.Timestamp(d).date() for d in existing['run_day']} if len(existing.columns) else set()
    return [m for m in earlier if m not in have] + [RUN_DATE]


failed_tenants = []
failed_months = []   # (tenant, run date) that raised an error
skipped_months = []  # (tenant, run date) without enough data to train on

for tenant_db, tz in tenants.items():
    print(f"Processing tenant: {tenant_db}")
    client = None

    try:
        if not tz:
            raise ValueError("no time zone set in tenant_properties.timezone_name or tenants.timezone")
        client = get_clickhouse_client(tenant_db)

        missing = [t for t in SOURCE_TABLES if not int(client.command(f'EXISTS TABLE {t}'))]
        if missing:
            raise ValueError(f"{missing} do not exist yet (source data not loaded)")

        # Creating the table if this tenant doesn't have it yet
        client.command(PREDICTIONS_DDL)

        run_dates = months_to_write(client)
        if len(run_dates) > 1:
            print(f"  Backfilling months with no rows yet: {[str(d) for d in run_dates[:-1]]}")

        tenant_failed = False
        for run_date in run_dates:
            try:
                customers, orders, refunds = load_tenant_data(client, tz, run_date)
                print(f"\n  [{run_date}] {len(customers)} customers, {len(orders)} orders, "
                      f"{len(refunds)} refunds found (time zone {tz})")

                if not len(customers) or not len(orders):
                    reason = (f"no customers created before {run_date}" if not len(customers)
                              else f"no orders between {ORDERS_START_DATE} and {run_date}")
                    print(f"  SKIPPED {run_date}: {reason}")
                    skipped_months.append((tenant_db, str(run_date)))
                    continue

                orders['net_spend'] = spend_as_of(orders, refunds, run_date, report=True)  # refunds known at run time

                train = build_training_set(customers, orders, refunds, run_date)
                reason = training_shortfall(train)
                if reason:
                    print(f"  SKIPPED {run_date}: {reason}")
                    skipped_months.append((tenant_db, str(run_date)))
                    continue
                print(f"  Training rows: {len(train):,} across {train['snapshot'].nunique()} snapshots")
                models = train_models(train)

                # Forecast from the run date itself, so the features describe the period being forecast
                predictions = predict(models, build_features(customers, orders, refunds, run_date))
                predictions = (predictions
                               .merge(actual_t0(orders, run_date), left_on='customer_system_shopify_id',
                                      right_index=True, how='left')
                               .fillna({'actual_total_spent_t0': 0, 'actual_orders_t0': 0}))
                predictions['actual_orders_t0'] = predictions['actual_orders_t0'].astype(int)
                predictions['run_date'] = pd.to_datetime(run_date)
                predictions = predictions[OUTPUT_COLS]
                print(f"  t0 actual: {predictions['actual_orders_t0'].sum():,} orders, "
                      f"spend {predictions['actual_total_spent_t0'].sum():,.0f}")

                write_predictions(client, run_date, predictions)
                print(f"  {run_date}: {len(predictions)} customers written")

            except Exception as exc:
                print(f"  FAILED {run_date}: {exc}")
                failed_months.append((tenant_db, str(run_date)))
                tenant_failed = True

        if tenant_failed:
            failed_tenants.append(tenant_db)
        else:
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
    if failed_months:
        print(f"Failed months ({len(failed_months)}): {failed_months}")
else:
    print("All tenants processed successfully.")
if skipped_months:
    print(f"Skipped months, not enough data ({len(skipped_months)}): {skipped_months}")

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
            COUNT(*) AS records
        FROM {TABLE}
        GROUP BY run_date
        ORDER BY run_date ASC
        """))
        print()
    finally:
        client.close()
