#!/usr/bin/env python
# Converted from 02_ds_cltv.ipynb. Run: python 02_ds_cltv.py

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

pipeline_metrics = PipelineMetrics("cltv")

import os
import warnings
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import psycopg2
import clickhouse_connect
import hvac
from hvac.exceptions import InvalidPath
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error, brier_score_loss

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
## List of tenants with Shopify connection status 'connected'

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
tenants = list(dict.fromkeys(tenants['schema_name'].tolist()))  # tenant_properties can hold duplicate rows (seen in prod)

print(f"Found {len(tenants)} tenants with Shopify connection status 'connected'.")
print(tenants)

# %%
## Set the run date to the first day of the current month

RUN_DATE = date.today().replace(day=1)
# RUN_DATE = datetime.strptime('2026-06-01', '%Y-%m-%d').date()  ## for ad-hoc runs

TABLE = 'ds_cltv'

# RUN_DATE is always (re)written. Any of the BACKFILL_MONTHS months before it
# that has no rows yet is written too, so a newly connected tenant starts with
# RUN_DATE plus 3 months of history, and a month that failed is retried next run.
BACKFILL_MONTHS = 3

# CLTV settings
HORIZON_DAYS = 90                 # prediction window, also the train/score gap
ORDERS_START_DATE = '2025-01-01'  # earliest order used for features and labels

# A month is skipped unless its training window has at least this many buyers
# and this many churners (CalibratedClassifierCV(cv=3) needs 3 of each class)
MIN_TRAINING_CLASS = 3

# Used only for tenants that don't have the table yet (same schema as the existing ones)
CLTV_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE}
(
    customer_system_shopify_id UInt64,
    churn_probability Float32,
    predicted_ltv_90d Float32,
    risk_tier String,
    run_date DateTime
)
ENGINE = MergeTree
ORDER BY customer_system_shopify_id
"""

# %%
## Reading customers and orders for a tenant

def load_tenant_data(client, run_date):
    """Customers created up to the day before `run_date` and their orders from
    ORDERS_START_DATE up to (not including) `run_date`, selecting only the
    columns the CLTV pipeline uses."""
    cut_off = run_date - timedelta(days=1)
    customers = client.query_df(f'''
        SELECT customer_system_shopify_id, created_at_date,
               email_marketing_consent_state, state
        FROM shopify_customers_transformed
        WHERE closed_at IS NULL
          AND created_at_date <= '{cut_off}'
    ''')

    # order_created_at is a timestamp, so the upper bound is "< run_date";
    # "<= cut_off" would compare against midnight and drop that whole day.
    orders = client.query_df(f'''
        SELECT customer_system_shopify_id, order_created_at, total_price,
               discount_amount, order_financial_status, order_source_name
        FROM shopify_orders_transformed
        WHERE closed_at IS NULL
          AND order_created_at >= '{ORDERS_START_DATE}'
          AND order_created_at < '{run_date}'
    ''')

    return customers, orders

# %%
## CLTV pipeline: 90-day churn probability + predicted LTV

"""
CLTV Pipeline — 90-day churn probability + predicted gross LTV.

For every customer, outputs:
    churn_probability : P(no purchase in the next `horizon_days`)
    predicted_ltv_90d : expected gross revenue in the next `horizon_days`

Temporal split: features are built strictly before an anchor date; labels from
the horizon after it. TRAIN anchor = SCORE anchor - horizon, so training labels
sit fully inside observed data while the score window is the unobserved future.
Monetary value is gross (all orders summed as-is, including voided/pending).
"""

def _coerce_dtypes(cust: pd.DataFrame, ord_: pd.DataFrame):
    """Normalize datetimes and align id dtypes before any join/reindex."""
    cust, ord_ = cust.copy(), ord_.copy()
    ord_["order_created_at"] = pd.to_datetime(ord_["order_created_at"])
    cust["created_at_date"] = pd.to_datetime(cust["created_at_date"])
    cust["customer_system_shopify_id"] = cust["customer_system_shopify_id"].astype("int64")
    ord_["customer_system_shopify_id"] = ord_["customer_system_shopify_id"].astype("int64")
    ord_["order_month"] = ord_["order_created_at"].dt.to_period("M")
    return cust, ord_


def build_features(cust: pd.DataFrame,
                   ord_: pd.DataFrame,
                   asof: pd.Timestamp,
                   channel_encoder: LabelEncoder | None = None):
    """One feature row per customer using only orders strictly before `asof`.
    Returns (features indexed by customer id over ALL customers, channel_encoder)."""
    win_90d_start = asof - timedelta(days=90)
    win_prior_start = asof - timedelta(days=180)

    hist = ord_[ord_["order_created_at"] < asof].copy()
    grp = hist.groupby("customer_system_shopify_id")

    rfm = grp["order_created_at"].agg(
        last_order_date="max", first_order_date="min", frequency="count",
    )
    rfm["recency_days"] = (asof - rfm["last_order_date"]).dt.days
    rfm["tenure_days"] = (asof - rfm["first_order_date"]).dt.days

    monetary = grp["total_price"].agg(total_revenue="sum", avg_order_value="mean")

    def _avg_gap(dates: pd.Series):
        s = dates.sort_values()
        if len(s) < 2:
            return np.nan
        return s.diff().dt.days.dropna().mean()

    avg_gap = grp["order_created_at"].apply(_avg_gap).rename("avg_days_between_orders")
    active_months = grp["order_month"].nunique().rename("num_active_months")

    o90 = hist[hist["order_created_at"] >= win_90d_start]
    opr = hist[(hist["order_created_at"] >= win_prior_start) &
               (hist["order_created_at"] < win_90d_start)]
    last_90d = o90.groupby("customer_system_shopify_id").agg(
        orders_last_90d=("total_price", "count"),
        revenue_last_90d=("total_price", "sum"),
    )
    prior_90d = opr.groupby("customer_system_shopify_id").agg(
        revenue_prior_90d=("total_price", "sum"),
    )

    # Velocity: trailing 30d vs the 30d before it (short-horizon trajectory).
    win_30d_start = asof - timedelta(days=30)
    win_prior30_start = asof - timedelta(days=60)
    o30 = hist[hist["order_created_at"] >= win_30d_start]
    opr30 = hist[(hist["order_created_at"] >= win_prior30_start) &
                 (hist["order_created_at"] < win_30d_start)]
    last_30d = o30.groupby("customer_system_shopify_id").agg(
        orders_last_30d=("total_price", "count"),
        revenue_last_30d=("total_price", "sum"),
    )
    prior_30d = opr30.groupby("customer_system_shopify_id").agg(
        orders_prior_30d=("total_price", "count"),
        revenue_prior_30d=("total_price", "sum"),
    )

    # Flags must be added to `hist` and regrouped (the first grp is now stale).
    hist["is_refunded"] = (hist["order_financial_status"] == "refunded").astype(int)
    hist["has_discount"] = (hist["discount_amount"].fillna(0) > 0).astype(int)
    grp2 = hist.groupby("customer_system_shopify_id")
    risk = grp2.agg(
        refund_rate=("is_refunded", "mean"),
        discount_usage_rate=("has_discount", "mean"),
        avg_discount_amount=("discount_amount", "mean"),
    )
    dominant_channel = (grp2["order_source_name"]
                        .agg(lambda x: x.value_counts().index[0] if len(x) else "unknown")
                        .rename("dominant_channel"))

    feats = (rfm
             .join(monetary, how="left")
             .join(avg_gap, how="left")
             .join(active_months, how="left")
             .join(last_90d, how="left")
             .join(prior_90d, how="left")
             .join(last_30d, how="left")
             .join(prior_30d, how="left")
             .join(risk, how="left")
             .join(dominant_channel, how="left"))

    for c in ["orders_last_90d", "revenue_last_90d", "revenue_prior_90d",
              "orders_last_30d", "revenue_last_30d",
              "orders_prior_30d", "revenue_prior_30d"]:
        feats[c] = feats[c].fillna(0)

    # Bounded momentum in [-1, 1]: stable when either window is small/zero,
    # never explodes on a tiny denominator (unlike a raw ratio).
    def _bounded(last, prior):
        denom = last + prior
        return np.where(denom > 0, (last - prior) / denom, 0.0)

    feats["revenue_trend"] = _bounded(feats["revenue_last_90d"], feats["revenue_prior_90d"])
    feats["revenue_velocity_30d"] = _bounded(feats["revenue_last_30d"], feats["revenue_prior_30d"])
    feats["order_velocity_30d"] = _bounded(feats["orders_last_30d"], feats["orders_prior_30d"])

    # Recency relative to the customer's OWN cadence: >1 means overdue vs their
    # typical gap (decelerating); <1 means ordering faster than usual.
    feats["recency_vs_gap"] = np.where(
        feats["avg_days_between_orders"] > 0,
        feats["recency_days"] / feats["avg_days_between_orders"], 0.0,
    )

    cust_idx = cust.set_index("customer_system_shopify_id")

    # Span ALL customers (never-ordered included), then fill.
    feats = feats.reindex(cust_idx.index)
    zero_cols = [
        "frequency", "total_revenue", "avg_order_value",
        "orders_last_90d", "revenue_last_90d", "revenue_prior_90d",
        "orders_last_30d", "revenue_last_30d", "orders_prior_30d", "revenue_prior_30d",
        "revenue_trend", "revenue_velocity_30d", "order_velocity_30d", "recency_vs_gap",
        "refund_rate", "discount_usage_rate",
        "avg_discount_amount", "num_active_months", "avg_days_between_orders",
    ]
    feats[zero_cols] = feats[zero_cols].fillna(0)

    # No orders before asof: recency/tenure default to account age.
    acct_age = (asof - cust_idx["created_at_date"]).dt.days.clip(lower=0)
    feats["recency_days"] = feats["recency_days"].fillna(acct_age)
    feats["tenure_days"] = feats["tenure_days"].fillna(acct_age)

    feats["marketing_consent"] = (
        cust_idx["email_marketing_consent_state"].reindex(feats.index).fillna("")
        == "subscribed").astype(int)
    feats["account_enabled"] = (
        cust_idx["state"].reindex(feats.index).fillna("") == "enabled").astype(int)

    # Shared channel encoder across train/score; unseen categories -> 'unknown'.
    chan = feats["dominant_channel"].fillna("unknown").astype(str)
    if channel_encoder is None:
        channel_encoder = LabelEncoder()
        channel_encoder.fit(pd.concat([chan, pd.Series(["unknown"])]))
    known = set(channel_encoder.classes_)
    chan = chan.where(chan.isin(known), "unknown")
    feats["dominant_channel_enc"] = channel_encoder.transform(chan)

    return feats, channel_encoder


def build_labels(ord_: pd.DataFrame, index: pd.Index, asof: pd.Timestamp,
                 horizon_days: int = 90):
    """Forward labels over (asof, asof + horizon]:
    churn_label = 1 if zero orders in window; ltv_label = gross revenue in window."""
    win_end = asof + timedelta(days=horizon_days)
    fwd = ord_[(ord_["order_created_at"] >= asof) & (ord_["order_created_at"] < win_end)]
    fwd_rev = (fwd.groupby("customer_system_shopify_id")["total_price"]
               .sum().reindex(index).fillna(0.0))
    churn_label = (fwd_rev <= 0).astype(int)
    return churn_label.rename("churn_label"), fwd_rev.rename("ltv_label")


FEATURE_COLS = [
    "recency_days", "frequency", "tenure_days",
    "total_revenue", "avg_order_value", "avg_days_between_orders",
    "num_active_months", "orders_last_90d", "revenue_last_90d",
    "revenue_prior_90d", "revenue_trend",
    "orders_last_30d", "revenue_last_30d", "revenue_velocity_30d",
    "order_velocity_30d", "recency_vs_gap",
    "refund_rate", "discount_usage_rate", "avg_discount_amount",
    "marketing_consent", "account_enabled", "dominant_channel_enc",
]


def run_cltv(cust: pd.DataFrame, ord_: pd.DataFrame,
             score_asof: pd.Timestamp | str,
             horizon_days: int = 90,
             drop_snapshot_features: bool = False):
    """Train on the observed horizon before `score_asof`, then score every
    customer for the `horizon_days` after it.

    drop_snapshot_features: exclude marketing_consent / account_enabled, which
        come from a live snapshot in `cust` (no timestamp) and may leak.
    """
    cust, ord_ = _coerce_dtypes(cust, ord_)
    score_asof = pd.Timestamp(score_asof)
    train_asof = score_asof - timedelta(days=horizon_days)

    print(f"Order date range : {ord_['order_created_at'].min()} -> {ord_['order_created_at'].max()}")
    print(f"TRAIN as-of      : {train_asof.date()} (labels {train_asof.date()}..{score_asof.date()})")
    print(f"SCORE as-of      : {score_asof.date()} (predicting next {horizon_days} days)\n")

    Xtr, enc = build_features(cust, ord_, train_asof)
    churn_y, ltv_y = build_labels(ord_, Xtr.index, train_asof, horizon_days)

    # Train only on customers who existed by train_asof.
    existed = (cust.set_index("customer_system_shopify_id")["created_at_date"]
               .reindex(Xtr.index) <= train_asof)

    model_cols = list(FEATURE_COLS)
    if drop_snapshot_features:
        model_cols = [c for c in model_cols
                      if c not in ("marketing_consent", "account_enabled")]
        print("Snapshot features dropped: marketing_consent, account_enabled\n")

    Xtr_fit = Xtr.loc[existed, model_cols].fillna(0)
    churn_fit, ltv_fit = churn_y.loc[existed], ltv_y.loc[existed]

    print(f"Training rows    : {len(Xtr_fit):,}")
    print(f"Churn base rate  : {churn_fit.mean():.3f} ({churn_fit.sum():,}/{len(churn_fit):,})")
    print(f"Mean 90d LTV     : {ltv_fit.mean():.2f}\n")

    rf_params = dict(n_estimators=300, max_depth=8, min_samples_leaf=5,
                     class_weight="balanced", random_state=42, n_jobs=-1)

    # --- Temporal validation -----------------------------------------------
    # Fit a probe on an EARLIER anchor (train_asof - horizon) and validate on
    # the train_asof window. This measures forward generalization across time
    # (the real deployment scenario) instead of same-period random holdout, and
    # surfaces seasonal drift. Falls back to a random split if history is too
    # short to carve an earlier fully-observed window.
    val_asof = train_asof - timedelta(days=horizon_days)
    earliest_order = ord_["order_created_at"].min()
    if val_asof - timedelta(days=180) >= earliest_order:
        Xv, _ = build_features(cust, ord_, val_asof, channel_encoder=enc)
        cy_v, _ = build_labels(ord_, Xv.index, val_asof, horizon_days)
        existed_v = (cust.set_index("customer_system_shopify_id")["created_at_date"]
                     .reindex(Xv.index) <= val_asof)
        Xv_fit = Xv.loc[existed_v, model_cols].fillna(0)
        yv = cy_v.loc[existed_v]
        auc_probe = RandomForestClassifier(**rf_params).fit(Xtr_fit, churn_fit)
        val_p = auc_probe.predict_proba(Xv_fit)[:, 1]
        print(f"Temporal val anchor     : {val_asof.date()} -> train {train_asof.date()}")
        print(f"Temporal ROC-AUC (churn): {roc_auc_score(yv, val_p):.3f}")
    else:
        Xa, Xb, ya, yb = train_test_split(
            Xtr_fit, churn_fit, test_size=0.25, random_state=42, stratify=churn_fit)
        auc_probe = RandomForestClassifier(**rf_params).fit(Xa, ya)
        val_p = auc_probe.predict_proba(Xb)[:, 1]
        print("Temporal val: insufficient history, using random holdout")
        print(f"Holdout ROC-AUC (churn) : {roc_auc_score(yb, val_p):.3f}")

    # Calibrated production classifier (probabilities feed the LTV product).
    churn_model = CalibratedClassifierCV(
        RandomForestClassifier(**rf_params), method="isotonic", cv=3,
    ).fit(Xtr_fit, churn_fit)
    cal_p = churn_model.predict_proba(Xtr_fit)[:, 1]
    print(f"Calibrated Brier        : {brier_score_loss(churn_fit, cal_p):.4f}")
    print(f"Mean P(churn) {cal_p.mean():.3f} (base rate {churn_fit.mean():.3f})\n")

    # --- Spend regressor on log revenue + Duan smearing --------------------
    # Revenue is right-skewed; squared-error on raw dollars chases the tail.
    # Train on log1p(revenue), then retransform with the smearing estimator:
    # E[rev] = expm1(pred) * mean(exp(residual)). The smear factor corrects the
    # downward bias of naive expm1 (Jensen). Buyers only; LTV = P(buy)*E[spend].
    buyers = ltv_fit > 0
    y_log = np.log1p(ltv_fit.loc[buyers])
    ltv_model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=5, learning_rate=0.05,
        l2_regularization=1.0, random_state=42,
    ).fit(Xtr_fit.loc[buyers], y_log)

    log_pred_tr = ltv_model.predict(Xtr_fit.loc[buyers])
    resid = y_log.values - log_pred_tr
    smear = float(np.mean(np.exp(resid)))   # Duan smearing factor

    def predict_spend(X):
        return np.expm1(ltv_model.predict(X)).clip(min=0) * smear

    mae = mean_absolute_error(ltv_fit.loc[buyers], predict_spend(Xtr_fit.loc[buyers]))
    print(f"Train MAE (LTV|buyers)  : {mae:.2f} (mean buyer spend {ltv_fit.loc[buyers].mean():.2f}, "
          f"smear {smear:.3f})")

    pred_ltv_tr = (1.0 - churn_model.predict_proba(Xtr_fit)[:, 1]) * predict_spend(Xtr_fit)
    print(f"Train mean pred LTV     : {pred_ltv_tr.mean():.2f} (actual {ltv_fit.mean():.2f})\n")

    # Score as of the real scoring date.
    Xsc, _ = build_features(cust, ord_, score_asof, channel_encoder=enc)
    Xsc_in = Xsc[model_cols].fillna(0)
    p_churn = churn_model.predict_proba(Xsc_in)[:, 1]
    spend_if_buy = predict_spend(Xsc_in)
    pred_ltv = (1.0 - p_churn) * spend_if_buy

    # --- Never-ordered customers: segmented + tenure-decayed first purchase --
    # P(buy) = seg_first_purchase_rate * exp(-days_since_signup / lambda)
    # LTV    = P(buy) * seg_first_order_value
    # Base rate and order value are segmented by marketing consent (subscribed
    # signups convert differently from non-subscribed), with a global fallback
    # for thin segments. lambda = median signup->first-order days among buyers.
    cust_idx = cust.set_index("customer_system_shopify_id")
    no_hist = (Xsc["frequency"].values == 0)
    tr_no_hist = (Xtr.loc[Xtr_fit.index, "frequency"] == 0)
    tr_fp = ltv_fit.loc[tr_no_hist]

    g_rate = (tr_fp > 0).mean() if tr_no_hist.any() else 0.0
    g_val = tr_fp[tr_fp > 0].mean() if (tr_fp > 0).any() else 0.0
    g_val = 0.0 if np.isnan(g_val) else g_val

    # Per-segment first-purchase stats keyed by consent (subscribed yes/no).
    consent_tr = (cust_idx["email_marketing_consent_state"].reindex(tr_fp.index).fillna("")
                  == "subscribed")
    seg_rate, seg_val = {}, {}
    for key, mask in [(1, consent_tr.values), (0, ~consent_tr.values)]:
        seg = tr_fp[mask]
        seg_rate[key] = (seg > 0).mean() if len(seg) >= 500 and (seg > 0).any() else g_rate
        sv = seg[seg > 0].mean() if (seg > 0).any() else g_val
        seg_val[key] = g_val if (len(seg) < 500 or np.isnan(sv)) else sv

    first_order = ord_.groupby("customer_system_shopify_id")["order_created_at"].min()
    days_to_first = ((first_order - cust_idx["created_at_date"].reindex(first_order.index))
                     .dt.days).clip(lower=0)
    lam = days_to_first.median()
    lam = 90.0 if (np.isnan(lam) or lam <= 0) else float(lam)

    consent_sc = (cust_idx["email_marketing_consent_state"].reindex(Xsc.index).fillna("")
                  == "subscribed").astype(int).values
    base_rate_vec = np.where(consent_sc == 1, seg_rate[1], seg_rate[0])
    base_val_vec = np.where(consent_sc == 1, seg_val[1], seg_val[0])

    dormancy = (score_asof - cust_idx["created_at_date"].reindex(Xsc.index)).dt.days
    dormancy = dormancy.clip(lower=0).fillna(0).values
    cold_p_buy = base_rate_vec * np.exp(-dormancy / lam)
    pred_ltv[no_hist] = (cold_p_buy * base_val_vec)[no_hist]
    p_churn[no_hist] = (1.0 - cold_p_buy)[no_hist]
    print(f"Cold-start (never-ordered): {no_hist.sum():,} customers")
    print(f"  rate subscribed {seg_rate[1]:.4f} / other {seg_rate[0]:.4f}, lambda {lam:.0f}d")
    print(f"  P(buy) {cold_p_buy[no_hist].min():.4f}..{cold_p_buy[no_hist].max():.4f}, "
          f"mean LTV {(cold_p_buy * base_val_vec)[no_hist].mean():.2f}\n")

    results = pd.DataFrame({
        "customer_system_shopify_id": Xsc.index,
        "recency_days": Xsc["recency_days"].values,
        "frequency": Xsc["frequency"].values,
        "total_revenue_hist": Xsc["total_revenue"].values,
        "churn_probability": p_churn,
        "predicted_ltv_90d": pred_ltv.round(2),
    })

    # Relative (percentile) risk tiers, not absolute risk.
    q = results["churn_probability"].rank(pct=True)
    results["risk_tier"] = pd.cut(
        q, bins=[0, 0.20, 0.80, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"], include_lowest=True)
    results = results.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    fi = pd.Series(auc_probe.feature_importances_, index=model_cols)
    print("Top churn drivers:")
    print(fi.sort_values(ascending=False).head(8).round(3).to_string(), "\n")

    print(f"Customers scored : {len(results):,}")
    print(results["risk_tier"].value_counts().sort_index().to_string())
    print(f"\nMean predicted 90d LTV : {results['predicted_ltv_90d'].mean():.2f}")
    print(f"  never-ordered ({(results['frequency'] == 0).sum():,}): "
          f"{results.loc[results['frequency'] == 0, 'predicted_ltv_90d'].mean():.2f} each")
    print(f"Total predicted 90d revenue : {results['predicted_ltv_90d'].sum():,.0f}")
    return results


def data_shortfall(cust: pd.DataFrame, ord_: pd.DataFrame,
                   run_date, horizon_days: int = 90):
    """Why `run_date` can't be scored, or None if it can. run_cltv trains on the
    customers who existed at train_asof = run_date - horizon, labelled by whether
    they bought in [train_asof, run_date) (the same labels as build_labels); the
    calibrated churn model needs MIN_TRAINING_CLASS buyers and churners."""
    if not len(cust):
        return "no customers"
    if not len(ord_):
        return "no orders"
    cust, ord_ = _coerce_dtypes(cust, ord_)
    score_asof = pd.Timestamp(run_date)
    train_asof = score_asof - timedelta(days=horizon_days)
    window = f"{train_asof.date()}..{score_asof.date()}"

    existed = cust.loc[cust["created_at_date"] <= train_asof, "customer_system_shopify_id"]
    if not len(existed):
        return f"no customers existed by the training anchor {train_asof.date()}"

    fwd = ord_[(ord_["order_created_at"] >= train_asof) &
               (ord_["order_created_at"] < score_asof) &
               ord_["customer_system_shopify_id"].isin(existed)]
    n_buyers = int((fwd.groupby("customer_system_shopify_id")["total_price"].sum() > 0).sum())
    n_churners = len(existed) - n_buyers
    if n_buyers < MIN_TRAINING_CLASS:
        return f"only {n_buyers} buyers in training window {window} (need {MIN_TRAINING_CLASS})"
    if n_churners < MIN_TRAINING_CLASS:
        return f"only {n_churners} churners in training window {window} (need {MIN_TRAINING_CLASS})"
    return None

# %%
## Scoring customers and writing CLTV for every connected tenant

UPLOAD_COLS = ['customer_system_shopify_id', 'churn_probability',
               'predicted_ltv_90d', 'risk_tier', 'run_date']


def write_cltv(client, run_date, results):
    """Replace the rows for `run_date` with `results`."""
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
        client.insert_df(TABLE, results)
        pipeline_metrics.add_rows(len(results))
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
skipped_months = []  # (tenant, run date) without enough data to score

for tenant_db in tenants:
    print(f"Processing tenant: {tenant_db}")
    client = None

    try:
        client = get_clickhouse_client(tenant_db)

        missing = [t for t in ('shopify_customers_transformed', 'shopify_orders_transformed')
                   if not int(client.command(f'EXISTS TABLE {t}'))]
        if missing:
            raise ValueError(f"{missing} do not exist yet (source data not loaded)")

        # Creating the table if this tenant doesn't have it yet
        client.command(CLTV_DDL)

        run_dates = months_to_write(client)
        if len(run_dates) > 1:
            print(f"  Backfilling months with no rows yet: {[str(d) for d in run_dates[:-1]]}")

        tenant_failed = False
        for run_date in run_dates:
            try:
                customers, orders = load_tenant_data(client, run_date)
                print(f"\n  [{run_date}] {customers.shape[0]} customers, {orders.shape[0]} orders found\n")

                reason = data_shortfall(customers, orders, run_date, HORIZON_DAYS)
                if reason:
                    print(f"  SKIPPED {run_date}: {reason}")
                    skipped_months.append((tenant_db, str(run_date)))
                    continue

                results = run_cltv(customers, orders, score_asof=run_date, horizon_days=HORIZON_DAYS)
                results['run_date'] = pd.to_datetime(run_date)
                results = results[UPLOAD_COLS]

                write_cltv(client, run_date, results)
                print(f"  {run_date}: {len(results)} customers written")

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
