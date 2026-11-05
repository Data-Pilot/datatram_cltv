#!/usr/bin/env python
# Converted from 03_ds_segmentation.ipynb. Run: python 03_ds_segmentation.py

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
DEFAULT_ENVIRONMENT = "dev"
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

pipeline_metrics = PipelineMetrics("segmentation")

import csv
import io
import os
import re
import warnings
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

import pandas as pd
import psycopg2
from psycopg2 import sql
import requests
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
query = '''
SELECT
    t.tenant_id,
    t.tenant_name,
    REPLACE(t.tenant_name, ' ', '_') || '_' || t.tenant_id AS schema_name,
    t.created_at,
    tp.source_connection_status AS shopify_connection_status
FROM "dev_datatram".tenants t
LEFT JOIN "dev_datatram".tenant_properties tp
    ON tp.tenant_id = t.tenant_id
   AND tp.source_id = (
        SELECT source_id
        FROM "dev_datatram".data_sources
        WHERE source_id = 1
   );
'''

tenants = pd.read_sql_query(query, conn)

conn.close()

tenants = tenants[tenants['shopify_connection_status'] == 'connected'].copy()
tenants = tenants['schema_name'].tolist()

print(f"Found {len(tenants)} tenants with Shopify connection status 'connected'.")
print(tenants)

# %%
## Set the run date to the first day of the current month

RUN_DATE = date.today().replace(day=1)
# RUN_DATE = datetime.strptime('2026-01-01', '%Y-%m-%d').date()  ## for ad-hoc runs

TABLE = 'ds_segmentation'

# RUN_DATE is always (re)written. Any of the BACKFILL_MONTHS months before it
# that has no rows yet is written too, so a newly connected tenant starts with
# RUN_DATE plus 3 months of history, and a month that failed is retried next run.
BACKFILL_MONTHS = 3

# Segmentation settings ("last month" = the calendar month before the run date)
TOP_PERCENTILE = 0.8   # high spenders / loyal = buyers at or above this percentile
LOYAL_MIN_ORDERS = 2   # a loyal customer has at least this many valid orders
ACTIVE_DAYS = 30       # active = a valid order within this many days before the run date

# Used only for tenants that don't have the table yet (same schema as the existing ones)
SEGMENTATION_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE}
(
    customer_system_shopify_id String,
    segment_unsubscribed UInt8,
    segment_high_spenders UInt8,
    segment_cart_abandoners UInt8,
    segment_one_timers UInt8,
    segment_indecisive_buyers UInt8,
    segment_idle_customers UInt8,
    segment_loyal_customers UInt8,
    run_date Date,
    active_customer UInt8 DEFAULT 0
)
ENGINE = ReplacingMergeTree
ORDER BY (run_date, customer_system_shopify_id)
"""

# %%
## Reading customers, orders and abandoned checkouts for a tenant

def select_df(client, columns, from_where):
    """SELECT `columns` as a DataFrame. clickhouse_connect returns a frame with
    no columns at all when nothing matches (e.g. a month with no abandoned
    checkouts), so an empty result is rebuilt with the expected columns."""
    df = client.query_df(f"SELECT {', '.join(columns)} {from_where}")
    return df if len(df.columns) else pd.DataFrame(columns=columns)


def load_tenant_data(client, run_date):
    """Customers created up to the day before `run_date`, plus their whole order
    and abandoned-checkout history up to (not including) `run_date`, selecting only
    the columns the segments use. closed_at IS NULL keeps the current version
    of each row (older versions are kept in the same table)."""
    cut_off = run_date - timedelta(days=1)
    customers = select_df(
        client,
        ['customer_system_shopify_id', 'email_marketing_consent_state', 'sms_marketing_consent_state'],
        f"""FROM shopify_customers_transformed
           WHERE closed_at IS NULL
             AND created_at_date <= '{cut_off}'""",
    )

    # order_created_at is a timestamp, so the upper bound is "< run_date";
    # "<= cut_off" would compare against midnight and drop that whole day.
    orders = select_df(
        client,
        ['customer_system_shopify_id', 'order_system_shopify_id', 'order_created_at',
         'order_fulfillment_status', 'current_total_price'],
        f"""FROM shopify_orders_transformed
           WHERE closed_at IS NULL
             AND order_created_at < '{run_date}'""",
    )

    # Checkouts never completed, as of their current version
    carts = select_df(
        client,
        ['customer_system_shopify_id', 'order_abandonment_system_shopify_id',
         'order_abandonment_created_at'],
        f"""FROM shopify_abandonment_orders_transformed
           WHERE closed_at IS NULL
             AND order_abandonment_completed_at IS NULL
             AND order_abandonment_created_at < '{run_date}'""",
    )

    return customers, orders, carts

# %%
## Per-customer features shared by all segments

def build_customer_features(customers, orders, carts, run_date):
    """One row per customer (every row of `customers`, in the same order) with
    the order and checkout aggregates the segments are built from.

    - Orders cover each customer's whole history up to run_date; only "valid"
      orders (with a fulfillment status) count towards spend and order counts.
    - Spend is current_total_price summed over valid orders. This deliberately
      differs from ds_actuals.total_spent (Shopify's lifetime figure) and from
      02's gross revenue.
    - "Last month" is the calendar month before run_date: [its 1st, run_date).
    - Checkouts without a customer id can't be attributed and drop out.
    Customers with no orders/checkouts get 0 spend and counts, and NaT dates."""
    ids = customers['customer_system_shopify_id']
    last_month_start = pd.Timestamp((run_date - timedelta(days=1)).replace(day=1))

    # Exclude orders with no fulfillment status (literal "None" string or actual NaN)
    status = orders['order_fulfillment_status']
    valid_orders = orders[status.notna() & (status != 'None')]
    valid_last_month = valid_orders[pd.to_datetime(valid_orders['order_created_at']) >= last_month_start]
    carts_last_month = carts[pd.to_datetime(carts['order_abandonment_created_at']) >= last_month_start]

    by_valid = valid_orders.groupby('customer_system_shopify_id')
    by_cart_last_month = carts_last_month.groupby('customer_system_shopify_id')

    features = pd.DataFrame({
        'total_spent': by_valid['current_total_price'].sum(),
        'valid_order_count': by_valid['order_system_shopify_id'].nunique(),
        'last_valid_order_at': by_valid['order_created_at'].max(),
        'valid_orders_last_month': (valid_last_month.groupby('customer_system_shopify_id')
                                    ['order_system_shopify_id'].nunique()),
        'abandoned_last_month': by_cart_last_month['order_abandonment_system_shopify_id'].nunique(),
        'last_abandoned_at': by_cart_last_month['order_abandonment_created_at'].max(),
        'abandoned_ever': (carts.groupby('customer_system_shopify_id')
                           ['order_abandonment_system_shopify_id'].nunique()),
    }).reindex(ids.values).reset_index(drop=True)

    features.insert(0, 'customer_system_shopify_id', ids.values)
    # Explicit dtypes so a tenant with no orders/checkouts (all-empty aggregates) still works
    features['total_spent'] = features['total_spent'].fillna(0.0).astype(float).round(2)
    for col in ('last_valid_order_at', 'last_abandoned_at'):
        features[col] = pd.to_datetime(features[col])
    count_cols = ['valid_order_count', 'valid_orders_last_month',
                  'abandoned_last_month', 'abandoned_ever']
    features[count_cols] = features[count_cols].fillna(0).astype('int64')

    # Not explicitly "subscribed" counts as unsubscribed - covers not_subscribed,
    # unsubscribed, and blank/NaN. fillna(False) is needed since nullable dtypes
    # propagate NA through .eq() instead of resolving to False.
    for channel in ('email', 'sms'):
        subscribed = customers[f'{channel}_marketing_consent_state'].eq('subscribed').fillna(False)
        features[f'{channel}_unsub'] = ~subscribed.values

    return features

# %%
## Segment definitions

SEGMENT_COLUMNS = [
    'segment_unsubscribed', 'segment_high_spenders', 'segment_cart_abandoners',
    'segment_one_timers', 'segment_indecisive_buyers', 'segment_idle_customers',
    'segment_loyal_customers', 'active_customer',
]


def build_segments(features, run_date):
    """0/1 flag per segment per customer, shaped for upload to ds_segmentation.

    Thresholds are relative to each tenant's own buyers, never fixed amounts,
    so the rules mean the same thing for any store size or currency:
    - "Buyers" are customers with at least one valid order. Percentiles are
      computed among buyers only, so customers who never ordered can't drag a
      threshold down to 0.
    - A tenant with no buyers gets a NaN threshold; comparisons against NaN are
      False, so it simply has no high spenders / loyal customers.

    The id is written as String and run_date as Date to match the existing
    ds_segmentation schema (both are sort-key columns, so their types can't be
    changed in place)."""
    f = features
    buyers = f['valid_order_count'] > 0
    spend_threshold = f.loc[buyers, 'total_spent'].quantile(TOP_PERCENTILE)
    loyal_threshold = f.loc[buyers, 'valid_order_count'].quantile(TOP_PERCENTILE)
    active_since = pd.Timestamp(run_date) - pd.Timedelta(days=ACTIVE_DAYS)
    # A valid order after the latest abandoned checkout means the customer came back and bought
    bought_after_abandoning = f['last_valid_order_at'] > f['last_abandoned_at']

    flags = pd.DataFrame({
        # Unsubscribed only if BOTH email and SMS are non-subscribed
        'segment_unsubscribed': f['email_unsub'] & f['sms_unsub'],
        # Top buyers by spend; zero spend never qualifies, even if the threshold is 0
        'segment_high_spenders': buyers & (f['total_spent'] > 0) & (f['total_spent'] >= spend_threshold),
        # Abandoned a checkout last month and hasn't bought since
        'segment_cart_abandoners': (f['abandoned_last_month'] > 0) & ~bought_after_abandoning,
        # Exactly one valid order, ever
        'segment_one_timers': f['valid_order_count'] == 1,
        # More abandoned checkouts than valid orders, both counted over last month
        'segment_indecisive_buyers': f['abandoned_last_month'] > f['valid_orders_last_month'],
        # Never bought and never even abandoned a checkout
        'segment_idle_customers': (f['valid_order_count'] == 0) & (f['abandoned_ever'] == 0),
        # Top buyers by order count; everyone tied at the threshold is included,
        # and a one-time buyer is never loyal
        'segment_loyal_customers': ((f['valid_order_count'] >= loyal_threshold)
                                    & (f['valid_order_count'] >= LOYAL_MIN_ORDERS)),
        # A valid order in the ACTIVE_DAYS before the run date
        'active_customer': f['last_valid_order_at'] >= active_since,
    })[SEGMENT_COLUMNS].astype('uint8')

    flags.insert(0, 'customer_system_shopify_id', f['customer_system_shopify_id'].astype(str))
    flags['run_date'] = pd.Timestamp(run_date).date()
    return flags


def print_segment_summary(segments):
    """Per-segment totals and the number of customers in no segment at all."""
    total = len(segments)
    no_segment = int((segments[SEGMENT_COLUMNS].sum(axis=1) == 0).sum())
    print(f"  Total customers segmented: {total:,}")
    print("  Total in each segment:")
    for name, count in segments[SEGMENT_COLUMNS].sum().sort_values(ascending=False).items():
        print(f"    {name:<27} {count:>8,}")
    print(f"  Customers with no segment at all: {no_segment:,} ({no_segment / total:.1%})")

# %%
## Segmenting customers and writing results for every connected tenant

SOURCE_TABLES = ('shopify_customers_transformed', 'shopify_orders_transformed',
                 'shopify_abandonment_orders_transformed')


def write_segments(client, run_date, segments):
    """Replace the rows for `run_date` with `segments`."""
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
        client.insert_df(TABLE, segments)
        pipeline_metrics.add_rows(len(segments))
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
skipped_months = []  # (tenant, run date) with no customers to segment

for tenant_db in tenants:
    print(f"Processing tenant: {tenant_db}")
    client = None

    try:
        client = get_clickhouse_client(tenant_db)

        missing = [t for t in SOURCE_TABLES if not int(client.command(f'EXISTS TABLE {t}'))]
        if missing:
            raise ValueError(f"{missing} do not exist yet (source data not loaded)")

        # Creating the table if this tenant doesn't have it yet
        client.command(SEGMENTATION_DDL)

        run_dates = months_to_write(client)
        if len(run_dates) > 1:
            print(f"  Backfilling months with no rows yet: {[str(d) for d in run_dates[:-1]]}")

        tenant_failed = False
        for run_date in run_dates:
            try:
                customers, orders, carts = load_tenant_data(client, run_date)
                print(f"  [{run_date}] {customers.shape[0]} customers, {orders.shape[0]} orders, "
                      f"{carts.shape[0]} abandoned checkouts found")

                if not len(customers):
                    print(f"  SKIPPED {run_date}: no customers created on or before {run_date - timedelta(days=1)}")
                    skipped_months.append((tenant_db, str(run_date)))
                    continue

                segments = build_segments(build_customer_features(customers, orders, carts, run_date), run_date)
                print_segment_summary(segments)

                write_segments(client, run_date, segments)
                print(f"  {run_date}: {len(segments)} customers written")

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
    print(f"Skipped months, no data ({len(skipped_months)}): {skipped_months}")

# %%
## Refresh Datatram/MinIO segment lists from ds_segmentation

LIST_TYPE = "data science"
UPLOAD_ENDPOINT = "/api/engagements/upload-list"
UPDATE_ENDPOINT = "/api/engagements/update-custom-list"
REQUEST_TIMEOUT_SECONDS = 300

EMAIL = "email"
FIRST_NAME = "first_name"
LAST_NAME = "last_name"
PHONE = "phone_number"

FIELDS = [EMAIL, FIRST_NAME, LAST_NAME, PHONE]
DEFAULT_HEADERS = {
    EMAIL: EMAIL,
    FIRST_NAME: FIRST_NAME,
    LAST_NAME: LAST_NAME,
    PHONE: PHONE,
}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

SEGMENT_LISTS = {
    "segment_unsubscribed": "Unsubscribed",
    "segment_high_spenders": "High Spenders",
    "segment_cart_abandoners": "Cart Abandoners",
    "segment_one_timers": "One Timers",
    "segment_indecisive_buyers": "Indecisive Buyers",
    "segment_idle_customers": "Idle Customers",
    "segment_loyal_customers": "Loyal Customers",
    "active_customer": "Active Customers",
}


DEFAULT_DATATRAM_BASE_URL = "https://dev.datatram.ai"

DATATRAM_BASE_URL_SECRET_CANDIDATES = (
    "datatram-base-url",
    "datatram-api-base-url",
    "datatram-api-url",
    "datatram-backend-url",
)


def get_datatram_base_url():
    value = os.getenv("DATATRAM_BASE_URL", "").strip()
    if value:
        return value.rstrip("/")

    for secret in DATATRAM_BASE_URL_SECRET_CANDIDATES:
        try:
            value = str(get_secret(vault, vault_mount, secret)).strip()
        except KeyError:
            continue
        if value:
            return value.rstrip("/")

    return DEFAULT_DATATRAM_BASE_URL.rstrip("/")


DATATRAM_BASE_URL = get_datatram_base_url()


def datatram_pg_conn():
    return psycopg2.connect(
        host=db_host,
        port=int(db_port),
        dbname=db_name,
        user=db_user,
        password=db_password,
        options=f"-c search_path={db_schema}",
    )


def normalize_list_name(name):
    return name.lower().replace(" ", "_")


def tenant_id_from_schema_name(schema_name):
    return int(str(schema_name).rsplit("_", 1)[1])


def resolve_api_key(conn, schema, tenant_id):
    query = sql.SQL("""
        SELECT u.user_id, k.hashed_api_key
        FROM {schema}.api_keys k
        JOIN {schema}.users u ON u.user_id = k.user_id
        WHERE u.tenant_id = %s
          AND u.is_active = TRUE
        ORDER BY (u.role_name = 'admin') DESC, k.created_at DESC
        LIMIT 1
    """).format(schema=sql.Identifier(schema))

    with conn.cursor() as cur:
        cur.execute(query, (tenant_id,))
        row = cur.fetchone()

    if not row:
        raise RuntimeError(f"No active API key found for tenant_id={tenant_id}")

    return int(row[0]), row[1]


def find_list(conn, schema, tenant_id, normalized_name):
    query = sql.SQL("""
        SELECT custom_list_id, list_type, email_address, first_name, last_name, phone_number
        FROM {schema}.custom_lists
        WHERE tenant_id = %s
          AND list_name = %s
        LIMIT 1
    """).format(schema=sql.Identifier(schema))

    with conn.cursor() as cur:
        cur.execute(query, (tenant_id, normalized_name))
        row = cur.fetchone()

    if not row:
        return None

    custom_list_id, list_type, email_h, first_h, last_h, phone_h = row

    headers = {
        EMAIL: email_h,
        FIRST_NAME: first_h,
        LAST_NAME: last_h,
        PHONE: phone_h,
    }

    return {
        "custom_list_id": int(custom_list_id),
        "list_type": list_type,
        "headers": {field: header for field, header in headers.items() if header},
    }


def clean_contacts(frame):
    seen = set()
    contacts = []

    for row in frame.to_dict("records"):
        email = str(row.get(EMAIL) or "").strip().lower()

        if not EMAIL_RE.match(email):
            continue

        if email in seen:
            continue

        seen.add(email)

        contacts.append({
            EMAIL: email,
            FIRST_NAME: "" if row.get(FIRST_NAME) is None else str(row.get(FIRST_NAME)).strip(),
            LAST_NAME: "" if row.get(LAST_NAME) is None else str(row.get(LAST_NAME)).strip(),
            PHONE: "" if row.get(PHONE) is None else str(row.get(PHONE)).strip(),
        })

    return contacts


def to_csv_bytes(contacts, headers):
    fields = [field for field in FIELDS if field in headers]

    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow([headers[field] for field in fields])

    for contact in contacts:
        writer.writerow([contact.get(field, "") for field in fields])

    return buffer.getvalue().encode("utf-8")


def post_form(endpoint, api_key, data, file_name, file_bytes):
    response = requests.post(
        DATATRAM_BASE_URL + endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        data=data,
        files={"file": (file_name, file_bytes, "text/csv")},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.ok:
        return response.json()

    detail = response.text
    try:
        detail = response.json().get("detail", detail)
    except ValueError:
        pass

    raise RuntimeError(f"HTTP {response.status_code}: {detail}")


def table_has_column(client, table, column):
    return bool(int(client.command(f"""
        SELECT count()
        FROM system.columns
        WHERE database = currentDatabase()
          AND table = '{table}'
          AND name = '{column}'
    """)))


def required_segment_tables_ready(client):
    missing = [
        table for table in (TABLE, 'ds_actuals')
        if not int(client.command(f"EXISTS TABLE {table}"))
    ]
    if missing:
        raise ValueError(f"{missing} do not exist yet; run ds_actuals and ds_segmentation first")

    if not table_has_column(client, 'ds_actuals', 'phone'):
        raise ValueError("ds_actuals has no phone column yet; run the updated ds_actuals notebook first")


def load_segment_contacts(client):
    required_segment_tables_ready(client)

    segment_selects = ",\n            ".join(
        f"toUInt8(s.{segment_col}) AS {segment_col}"
        for segment_col in SEGMENT_LISTS
    )

    return client.query_df(f"""
        SELECT
            toString(s.customer_system_shopify_id) AS customer_system_shopify_id,
            a.email AS email,
            a.first_name AS first_name,
            a.last_name AS last_name,
            a.phone AS phone_number,
            {segment_selects}
        FROM {TABLE} s
        LEFT JOIN ds_actuals a
            ON toString(a.customer_system_shopify_id) = toString(s.customer_system_shopify_id)
           AND toDate(a.run_date) = toDate(s.run_date)
        WHERE toDate(s.run_date) = toDate('{RUN_DATE}')
    """)


def create_list(api_key, tenant_id, user_id, list_name, contacts, headers, file_name, file_bytes):
    data = [
        ("tenant_id", str(tenant_id)),
        ("user_id", str(user_id)),
        ("list_name", list_name),
        ("list_type", LIST_TYPE),
        ("members", str(len(contacts))),
        ("email_address", headers[EMAIL]),
        ("first_name", headers.get(FIRST_NAME, "")),
        ("last_name", headers.get(LAST_NAME, "")),
        ("phone_number", headers.get(PHONE, "")),
        ("tags", "data-science"),
        ("tags", "auto-refresh"),
    ]

    return post_form(UPLOAD_ENDPOINT, api_key, data, file_name, file_bytes)


def replace_list(api_key, tenant_id, custom_list_id, file_name, file_bytes):
    data = [
        ("tenant_id", str(tenant_id)),
        ("custom_list_id", str(custom_list_id)),
    ]

    return post_form(UPDATE_ENDPOINT, api_key, data, file_name, file_bytes)


def refresh_segment_lists_for_tenant(tenant_db):
    tenant_id = tenant_id_from_schema_name(tenant_db)

    client = get_clickhouse_client(tenant_db)

    try:
        segment_contacts = load_segment_contacts(client)
    finally:
        client.close()

    if segment_contacts.empty:
        print(f"{tenant_db}: no ds_segmentation rows for {RUN_DATE}; skipping")
        return

    with datatram_pg_conn() as pg_conn:
        pg_conn.set_session(readonly=True)

        user_id, api_key = resolve_api_key(pg_conn, db_schema, tenant_id)

        for segment_col, list_name in SEGMENT_LISTS.items():
            existing = find_list(
                pg_conn,
                db_schema,
                tenant_id,
                normalize_list_name(list_name),
            )

            members = segment_contacts[
                segment_contacts[segment_col].astype(int) == 1
            ]

            contacts = clean_contacts(members)

            if len(members) > 0 and not contacts:
                print(f"{tenant_db}: {list_name} has members but no valid emails; keeping existing list")
                continue

            if existing and existing["list_type"] != LIST_TYPE:
                print(f"{tenant_db}: {list_name} is not a data-science list; skipping")
                continue

            if existing and EMAIL not in existing["headers"]:
                print(f"{tenant_db}: {list_name} has no email mapping; skipping")
                continue

            if not existing and not contacts:
                print(f"{tenant_db}: {list_name} is empty and does not exist yet; skipping create")
                continue

            headers = existing["headers"] if existing else DEFAULT_HEADERS
            csv_bytes = to_csv_bytes(contacts, headers)
            file_name = f"{normalize_list_name(list_name)}.csv"

            if existing:
                replace_list(
                    api_key=api_key,
                    tenant_id=tenant_id,
                    custom_list_id=existing["custom_list_id"],
                    file_name=file_name,
                    file_bytes=csv_bytes,
                )

                print(f"{tenant_db}: refreshed {list_name} with {len(contacts)} contacts")

            else:
                create_list(
                    api_key=api_key,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    list_name=list_name,
                    contacts=contacts,
                    headers=headers,
                    file_name=file_name,
                    file_bytes=csv_bytes,
                )

                print(f"{tenant_db}: created {list_name} with {len(contacts)} contacts")


failed_list_refreshes = []

tenant_names = tenants.keys() if isinstance(tenants, dict) else tenants

for tenant_db in tenant_names:
    print(f"Refreshing MinIO list uploads for tenant: {tenant_db}")

    try:
        refresh_segment_lists_for_tenant(tenant_db)
    except Exception as exc:
        print(f"ERROR refreshing lists for {tenant_db}: {exc}")
        failed_list_refreshes.append(tenant_db)


print("\nMinIO list refresh done.")

if failed_list_refreshes:
    print(f"Failed list refreshes ({len(failed_list_refreshes)}): {failed_list_refreshes}")
else:
    print("All MinIO segment lists refreshed successfully.")

pipeline_metrics.finish(success=not failed_tenants and not failed_list_refreshes)

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
