[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-3.x-017CEE?style=flat&logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-yellow.svg?style=flat)](LICENSE)

# Kairos

Kairos is a predictive data placement service for [Apache Airflow](https://airflow.apache.org/) workflows whose data lives in [Onedata](https://onedata.org/). It watches which files the tasks of a DAG read and write, predicts the next executions, and replicates the data to the Onedata provider where it will be needed before the task starts, balancing access latency against storage cost.

Each cycle Kairos:

1. ingests the file accesses of finished tasks from Prometheus into a local SQLite database,
2. predicts the next DAG/task executions and the providers they will run against,
3. plans and schedules replications and evictions across the Onedata providers, using measured provider latency and bandwidth.

## Repository layout

| Path | Content |
|---|---|
| `src/dag_monitoring` | Reads Onedata access metrics from Prometheus and stores them in SQLite |
| `src/dag_prediction` | Execution predictors (SETAR trees, global/space models, fallback model) |
| `src/data_placement` | Placement interface, the Kairos strategy and the Zenoh interrupt listener |
| `src/database` | SQLite persistence |
| `src/network_monitoring` | Provider latency and bandwidth measurement |
| `src/rest_api` | Onedata (`OnedataDataManager`) and Airflow (`AirflowManager`) REST clients |
| `src/runtime` | Configuration, CLI and the ingestion → prediction → placement pipeline |
| `deployment` | Kubernetes manifest templates |
| `tests` | Unit tests and smoke tests against live services |

---

## Running Kairos on a new cluster

### 1. Prerequisites

Kairos does not ship the services it observes. The cluster must already provide:

**Apache Airflow 3.x**
- The REST API server must be reachable from the Kairos pod, together with a user that can list DAGs and tasks (Kairos requests a token from `/auth/token`).

**Prometheus with the OTLP receiver enabled**
- Start Prometheus with `--web.enable-otlp-receiver`.
- Airflow task pods must push their metrics to it:
  ```bash
  OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://<prometheus-service>:9090/api/v1/otlp/v1/metrics
  OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
  ```

**Instrumented workloads**

Kairos only sees what the tasks report. Tasks must access Onedata through [onedatarestfsspec](https://github.com/bkryza/onedatarestfsspec), which emits the metrics Kairos queries:

| Metric | Required labels |
|---|---|
| `onedata_read_duration_seconds` (`_sum`) | `provider_id`, `session_id` |
| `onedata_write_duration_seconds` (`_sum`) | `provider_id`, `session_id` |

`session_id` must have the form `dag_id:dag_run_id:task_id:task_try`. Accesses without it are stored without a DAG/task and cannot be predicted.

**Onedata**
- A space holding the workflow data, supported by at least two providers (otherwise there is nothing to place).
- A second, empty space (the "devnull" space) that Kairos uses for latency and bandwidth probing.
- An access token for the Onezone and one for the providers, both with REST access to these spaces. Create them in the Onezone UI under *Tokens*.

**Zenoh (only for task-level optimization)**
- With `OPTIMIZATION_LEVEL=task`, Kairos reacts to task-completion interrupts published as JSON (`{"dag_id": ..., "task_id": ..., "task_start": ...}`) on the Zenoh key `airflow_interrupts`.
- Multicast scouting does not reach other pods, so in Kubernetes Kairos and the publisher connect to a Zenoh router (`deployment/zenoh_router.template`), set with `ZENOH_CONNECT`. Without it, Kairos discovers peers via multicast.
- With the default `OPTIMIZATION_LEVEL=dag`, no interrupts are needed.

### 2. Configure provider prices

Kairos weighs storage cost per provider. Create the price file from the example and enter the Onedata provider IDs of your space (find them in the Onezone UI or via `GET /api/v3/onezone/user/spaces/<space-id>`):

```bash
cp src/data_placement/strategies/price_conf.example.json src/data_placement/strategies/price_conf.json
```

`price_conf.json` is git-ignored and is baked into the image in the next step. To use another file, set `KAIROS_PRICE_CONFIG` to its path inside the container.

Optionally, fix the network values instead of measuring them. The probes take several minutes at startup and every `PAIRWISE_LATENCY_REFRESH_CYCLES` cycles, slow the workload down while they run, and vary strongly between runs. `src/network_monitoring/network_conf.json` holds the medians measured on the evaluation cluster; adjust it to your providers and set `KAIROS_NETWORK_CONFIG` to its path.

### 3. Build and push the image

```bash
export IMAGE=<registry>/<user>/kairos:latest
docker build -t "$IMAGE" .
docker push "$IMAGE"
```

### 4. Fill in the Kubernetes manifests

Every `*.template` in `deployment/` has a matching `*.yaml` you create from it. All `deployment/*.yaml` files are git-ignored, so the real secrets never end up in the repository.

```bash
cd deployment
for f in secret configmap persistent_volume_claim serviceaccount kustomization worker_job; do
  cp "$f.template" "$f.yaml"
done
```

| File | What to set |
|---|---|
| `secret.yaml` | `space-id`, `token-provider`, `token-zone`, `devnull-space-id`, `airflow-username`, `airflow-password`, and `airflow-api-service` as a full URL (e.g. `http://<airflow-api-server-service>:8080`) |
| `configmap.yaml` | `PROMETHEUS_HOST` (e.g. `http://<prometheus-service>:9090`), `ONEDATA_ONEZONE_HOST` (hostname without `https://`), optionally `OPTIMIZATION_LEVEL` (`dag` or `task`), `MONITOR_LOOKBACK`, `PAIRWISE_LATENCY_REFRESH_CYCLES`, `KAIROS_NETWORK_CONFIG` |
| `worker_job.yaml` | `image:` → your `$IMAGE` |
| `persistent_volume_claim.yaml` | `storageClassName` if the cluster has no default StorageClass |
| all files | `namespace:` if you do not deploy to `default` |

The SQLite database is stored at `/data/kairos.sqlite3` on the persistent volume, so the prediction history survives restarts.

### 5. Deploy

```bash
kubectl apply -k deployment/                # secret, configmap, PVC, service account + RBAC
kubectl apply -f deployment/worker_job.yaml # Kairos worker (Deployment)
kubectl rollout status deployment/predictor-placement-worker
```

To roll out a new image, push it and run `kubectl rollout restart deployment/predictor-placement-worker`.

### 6. Check that it works

```bash
kubectl logs deployment/predictor-placement-worker -f
```

On startup Kairos logs that the configuration, the SQLite database and the Onedata managers were initialized, then measures the provider latencies and enters its placement loop. Once a DAG has run, its accesses should appear in the ingestion logs.

| Symptom | Likely cause |
|---|---|
| `Missing Onezone host` / `Missing Onedata token` | `configmap.yaml` or `secret.yaml` not filled in or not applied |
| No accesses are ingested | Tasks do not use onedatarestfsspec, OTLP export to Prometheus is not configured, or `session_id` has the wrong format |
| `FileNotFoundError` for `price_conf.json` | Step 2 was skipped before building the image |
| Airflow authentication fails | `airflow-api-service` is not a full URL, or the credentials are wrong |
| Pod stays `Pending` | The PVC cannot bind. Check `kubectl describe pvc kairos-sqlite-pvc` |

---

## Configuration reference

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `PROMETHEUS_HOST` | yes | | Prometheus base URL |
| `ONEDATA_ONEZONE_HOST` | yes | | Onezone hostname |
| `ONEDATA_SPACE_ID` | yes | | Space whose data Kairos places |
| `ONEDATA_TOKEN_ZONE` | yes | | Onezone access token |
| `ONEDATA_TOKEN_PROVIDER` | yes | | Provider access token |
| `DEVNULL_SPACE_ID` | yes | | Empty space used for latency/bandwidth probing |
| `AIRFLOW_API_SERVICE` | yes | | Airflow API base URL |
| `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | yes | | Airflow credentials |
| `DATABASE_URL` | no | `/data/kairos.sqlite3` | SQLite database path (`--db-path` takes precedence) |
| `OPTIMIZATION_LEVEL` | no | `dag` | `dag` plans per DAG run, `task` also reacts to task interrupts |
| `ZENOH_CONNECT` | no | | Zenoh router endpoint for the task interrupts, e.g. `tcp/zenoh-router:7447`; empty = multicast scouting |
| `MONITOR_LOOKBACK` | no | `5` | Minutes of Prometheus history queried per cycle |
| `PAIRWISE_LATENCY_REFRESH_CYCLES` | no | `10` | Cycles between provider-to-provider measurements, `0` = never |
| `KAIROS_PRICE_CONFIG` | no | `src/data_placement/strategies/price_conf.json` | Provider price file |
| `KAIROS_REPLICATION_OVERHEAD_SECONDS` | no | `10` | Fixed Onedata replication overhead used when no probe transfer was measured |
| `KAIROS_REPLICATION_BANDWIDTH` | no | `20000000` | Bytes/s assumed for the data part of a replication |
| `KAIROS_NETWORK_CONFIG` | no | | JSON file with fixed provider latencies and replication overhead/bandwidth (e.g. `src/network_monitoring/network_conf.json`). When set, Kairos does not probe the network |
| `KAIROS_REPLICA_PROTECTION_SECONDS` | no | `300` | How long a replica Kairos created is kept from eviction after its task's predicted start |

Command-line options (`python -m src --help`): `--prometheus-host`, `--db-path` (overrides `DATABASE_URL`), `--lookback`, `--no-duration`, `--max-predictions`, `--log-level`.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/unit
```

The smoke tests in `tests/smoke` run against live services and need the environment variables above.

To run Kairos locally against a cluster, port-forward Prometheus and Airflow, export the variables above and run:

```bash
python -m src --db-path data/kairos.sqlite3 --log-level DEBUG
```

## License

Kairos is licensed under the GNU General Public License v3.0, see [LICENSE](LICENSE). Third-party components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
