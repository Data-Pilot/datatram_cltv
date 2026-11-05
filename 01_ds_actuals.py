#!/usr/bin/env python
# Converted from 01_ds_actuals.ipynb. Run: python 01_ds_actuals.py

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

pipeline_metrics = PipelineMetrics("actuals")

import psycopg2
import pandas as pd
import clickhouse_connect
from dotenv import load_dotenv
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# %%
## Reading ClickHouse + Postgres credentials from Vault

import os
import hvac
from hvac.exceptions import InvalidPath

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
# RUN_DATE = datetime.strptime('2026-06-01', '%Y-%m-%d').date()  ## for ad-hoc runs

TABLE = 'ds_actuals'

# RUN_DATE is always (re)written. Any of the BACKFILL_MONTHS months before it
# that has no rows yet is written too, so a newly connected tenant starts with
# RUN_DATE plus 3 months of history, and a month that failed is retried next run.
BACKFILL_MONTHS = 3

# Used only for tenants that don't have the table yet (same schema as the existing ones)
ACTUALS_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE}
(
    customer_system_shopify_id UInt64,
    first_name String,
    last_name String,
    email String,
    phone String,
    orders_count UInt32,
    total_spent Float32,
    created_at_date DateTime,
    run_date DateTime
)
ENGINE = MergeTree
ORDER BY customer_system_shopify_id
"""

# %%
## Reading customers and writing actuals for every connected tenant

ACTUALS_OUTPUT_COLS = [
    'customer_system_shopify_id',
    'first_name',
    'last_name',
    'email',
    'phone',
    'orders_count',
    'total_spent',
    'created_at_date',
    'run_date',
]


def table_has_column(client, table, column):
    return bool(int(client.command(f"""
        SELECT count()
        FROM system.columns
        WHERE database = currentDatabase()
          AND table = '{table}'
          AND name = '{column}'
    """)))


def source_phone_expression(client):
    """Return a ClickHouse SELECT expression for the customer phone field."""
    if table_has_column(client, 'shopify_customers_transformed', 'phone'):
        return 'phone'
    if table_has_column(client, 'shopify_customers_transformed', 'phone_number'):
        return 'phone_number AS phone'
    print("  WARNING: shopify_customers_transformed has no phone/phone_number column; writing blank phone values.")
    return "'' AS phone"


def ensure_actuals_schema(client):
    """Create ds_actuals if needed and add phone to existing tenant tables."""
    client.command(ACTUALS_DDL)
    client.command(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS phone String AFTER email")


def load_actuals(client, run_date):
    """Open customers created up to the day before `run_date`, stamped with `run_date`."""
    cut_off = run_date - timedelta(days=1)
    phone_expr = source_phone_expression(client)
    customers = client.query_df(f'''SELECT customer_system_shopify_id, first_name, last_name, email, {phone_expr}, orders_count, total_spent, created_at_date
                                FROM shopify_customers_transformed
                                WHERE closed_at IS NULL
                                AND created_at_date <= '{cut_off}' ''')
    customers['phone'] = customers['phone'].fillna('').astype(str)
    customers['run_date'] = pd.to_datetime(run_date)
    return customers[ACTUALS_OUTPUT_COLS]


def write_actuals(client, run_date, customers):
    """Replace the rows for `run_date` with `customers`."""
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
        client.insert_df(TABLE, customers)
        pipeline_metrics.add_rows(len(customers))
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
skipped_months = []  # (tenant, run date) with no customers to write

for tenant_db in tenants:
    print(f"Processing tenant: {tenant_db}")
    client = None

    try:
        client = get_clickhouse_client(tenant_db)

        if not int(client.command('EXISTS TABLE shopify_customers_transformed')):
            raise ValueError("shopify_customers_transformed does not exist yet (source data not loaded)")

        ensure_actuals_schema(client)

        run_dates = months_to_write(client)
        if len(run_dates) > 1:
            print(f"  Backfilling months with no rows yet: {[str(d) for d in run_dates[:-1]]}")

        tenant_failed = False
        for run_date in run_dates:
            try:
                customers = load_actuals(client, run_date)
                if not len(customers):
                    print(f"  SKIPPED {run_date}: no customers created on or before {run_date - timedelta(days=1)}")
                    skipped_months.append((tenant_db, str(run_date)))
                    continue

                write_actuals(client, run_date, customers)
                print(f"  {run_date}: {customers.shape[0]} customers written")

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
