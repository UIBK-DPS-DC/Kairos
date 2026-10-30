import pytest

from src.rest_api.onedata.onedata_helpers import Transfer, TransferStatus, TransferType
from src.rest_api.onedata.onedatamanager import OnedataDataManager


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text="", ok=True):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.ok = ok

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http error {self.status_code}")


@pytest.fixture
def manager():
    cfg = OnedataDataManager.load_config(
        token="dummy-token",
        onezone_host="onezone.example.org",
        oneprovider_host="provider.example.org",
        dev_mode=True,
    )
    return OnedataDataManager.build_storage_manager(
        config=cfg,
        provider_url="https://provider.example.org",
        space_id="space-1",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("guid#abc", "abc"),
        ("ch#abc", "abc"),
        ("plain-guid", "plain-guid"),
    ],
)
def test_extract_guid_supported_formats(raw, expected):
    assert OnedataDataManager.extract_guid(raw) == expected


@pytest.mark.unit
def test_infer_token_scope_posts_expected_payload_and_returns_json(
    monkeypatch, manager
):
    captured = {"url": None, "json": None, "timeout": None}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse(
            status_code=200,
            json_data={"tokenCaveats": [{"type": "interface", "interface": "rest"}]},
        )

    monkeypatch.setattr(manager._session, "post", fake_post)

    result = manager.infer_token_scope()

    assert captured["url"].endswith("/api/v3/onezone/tokens/infer_access_token_scope")
    assert captured["json"] == {"token": "dummy-token"}
    assert captured["timeout"] == (10, 30)
    assert result["tokenCaveats"][0]["interface"] == "rest"


@pytest.mark.unit
def test_diagnose_token_parses_caveats(monkeypatch, manager):
    def fake_infer_token_scope():
        return {
            "tokenCaveats": [
                {"type": "interface", "interface": "rest"},
                {"type": "service", "service": "opw-provider-1"},
                {"type": "ip", "whitelist": "10.0.0.0/24"},
            ]
        }

    monkeypatch.setattr(manager, "infer_token_scope", fake_infer_token_scope)

    result = manager.diagnose_token()

    assert result["valid"] is True
    assert result["has_rest"] is True
    assert result["interfaces"] == ["rest"]
    assert result["service_ids"] == ["opw-provider-1"]
    assert result["ip_ranges"] == ["10.0.0.0/24"]
    assert result["error"] == ""


@pytest.mark.unit
def test_diagnose_token_returns_error_when_infer_fails(monkeypatch, manager):
    def fake_infer_token_scope():
        raise RuntimeError("boom")

    monkeypatch.setattr(manager, "infer_token_scope", fake_infer_token_scope)

    result = manager.diagnose_token()

    assert result["valid"] is False
    assert result["has_rest"] is False
    assert result["error"] == "boom"


@pytest.mark.unit
def test_mint_rest_token_posts_expected_caveats_and_returns_token(monkeypatch, manager):
    captured = {"url": None, "body": None, "timeout": None}

    providers = [
        type("P", (), {"provider_id": "pid-a"})(),
        type("P", (), {"provider_id": "pid-b"})(),
    ]

    def fake_raw_get(url, params=None):
        _ = (params,)
        if url.endswith("/user/tokens/named"):
            return {"tokens": []}
        raise AssertionError(f"Unexpected URL in test: {url}")

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["body"] = json
        captured["timeout"] = timeout
        return DummyResponse(
            status_code=201, json_data={"token": "minted-token"}, ok=True
        )

    monkeypatch.setattr(manager, "_raw_get", fake_raw_get)
    monkeypatch.setattr(manager, "list_providers", lambda: providers)
    monkeypatch.setattr(manager._session, "post", fake_post)

    token = manager.mint_rest_token(name="unit-test-token")

    assert token == "minted-token"
    assert captured["url"].endswith("/api/v3/onezone/user/tokens/named")
    assert captured["timeout"] == (10, 30)
    caveats = captured["body"]["caveats"]
    assert {"type": "interface", "interface": "rest"} in caveats
    service_caveat = next(c for c in caveats if c.get("type") == "service")
    assert service_caveat["whitelist"] == ["ozw-onezone", "opw-pid-a", "opw-pid-b"]


@pytest.mark.unit
def test_mint_rest_token_deletes_existing_named_token(monkeypatch, manager):
    seen = {"delete_url": None}

    def fake_raw_get(url, params=None):
        _ = params
        if url.endswith("/user/tokens/named"):
            return {"tokens": ["tid-1"]}
        if url.endswith("/user/tokens/named/tid-1"):
            return {"name": "same-name"}
        raise AssertionError(f"Unexpected URL in test: {url}")

    def fake_delete(url, timeout):
        seen["delete_url"] = url
        return DummyResponse(status_code=204)

    def fake_post(url, json, timeout):
        _ = (url, json, timeout)
        return DummyResponse(
            status_code=201, json_data={"token": "minted-token"}, ok=True
        )

    monkeypatch.setattr(manager, "_raw_get", fake_raw_get)
    monkeypatch.setattr(manager, "list_providers", lambda: [])
    monkeypatch.setattr(manager._session, "delete", fake_delete)
    monkeypatch.setattr(manager._session, "post", fake_post)

    token = manager.mint_rest_token(name="same-name")

    assert token == "minted-token"
    assert seen["delete_url"].endswith("/api/v3/onezone/user/tokens/named/tid-1")


@pytest.mark.unit
def test_mint_rest_token_raises_runtime_error_on_bad_request(monkeypatch, manager):
    def fake_raw_get(url, params=None):
        _ = params
        if url.endswith("/user/tokens/named"):
            return {"tokens": []}
        raise AssertionError(f"Unexpected URL in test: {url}")

    def fake_post(url, json, timeout):
        _ = (url, json, timeout)
        return DummyResponse(
            status_code=400, json_data={"description": "invalid caveat"}, ok=False
        )

    monkeypatch.setattr(manager, "_raw_get", fake_raw_get)
    monkeypatch.setattr(manager, "list_providers", lambda: [])
    monkeypatch.setattr(manager._session, "post", fake_post)

    with pytest.raises(RuntimeError):
        manager.mint_rest_token(name="bad-request")


@pytest.mark.unit
def test_ensure_rest_token_updates_config_and_session(monkeypatch, manager):
    old_config = manager.config
    sentinel_session = object()

    monkeypatch.setattr(
        manager,
        "mint_rest_token",
        lambda name="onedata-manager-rest-token": "new-rest-token",
    )
    monkeypatch.setattr(manager, "_build_session", lambda: sentinel_session)

    updated = manager.ensure_rest_token(name="new-name")

    assert updated is True
    assert manager.config.token == "new-rest-token"
    assert manager.config.onezone_url == old_config.onezone_url
    assert manager.config.oneprovider_url == old_config.oneprovider_url
    assert manager._session is sentinel_session


@pytest.mark.unit
def test_get_replica_distribution_modern_api_computes_missing_percentage(
    monkeypatch, manager
):
    def fake_raw_get(url, params=None):
        _ = params
        assert url.endswith("/api/v3/oneprovider/data/file-1/distribution")
        return {
            "distributionPerProvider": {
                "pid-a": {"blocksPercentage": 25.0, "virtualSize": 100},
                "pid-b": {
                    "virtualSize": 100,
                    "distributionPerStorage": {"s1": {"blocks": [[0, 10], [10, 40]]}},
                },
            }
        }

    monkeypatch.setattr(manager, "_raw_get", fake_raw_get)

    dist = manager.get_replica_distribution("file-1")

    assert dist.file_ref == "file-1"
    assert dist.blocks_percentage("pid-a") == 25.0
    assert dist.blocks_percentage("pid-b") == 50.0
    assert set(dist.provider_ids) == {"pid-a", "pid-b"}
    assert dist.replica_count == 2


@pytest.mark.unit
def test_get_replica_distribution_legacy_mode_maps_blocks_to_percentages(monkeypatch):
    cfg = OnedataDataManager.load_config(
        token="dummy-token",
        onezone_host="onezone.example.org",
        oneprovider_host="provider.example.org",
        dev_mode=True,
    )
    manager = OnedataDataManager.build_storage_manager(
        config=cfg,
        provider_url="https://provider.example.org",
        space_id="space-1",
        use_guid=False,
    )

    def fake_raw_get(url, params=None):
        _ = (url, params)
        return {
            "replicasPerProvider": {
                "pid-a": [[0, 10]],
                "pid-b": [],
            }
        }

    monkeypatch.setattr(manager, "_raw_get", fake_raw_get)

    dist = manager.get_replica_distribution("legacy-file")

    assert dist.blocks_percentage("pid-a") == 100.0
    assert dist.blocks_percentage("pid-b") == 0.0


@pytest.mark.unit
def test_replicate_to_provider_builds_payload_and_returns_transfer(
    monkeypatch, manager
):
    calls = {"url": None, "body": None}

    def fake_raw_post(url, params=None, body=None):
        _ = params
        calls["url"] = url
        calls["body"] = body
        return {"transferId": "tr-repl-1"}

    monkeypatch.setattr(manager, "_raw_post", fake_raw_post)

    transfer = manager.replicate_to_provider(
        file_ref="guid#abc123",
        destination_provider_id="dst-provider",
        callback_url="https://callback.example",
    )

    assert calls["url"].endswith("/api/v3/oneprovider/transfers")
    assert calls["body"]["type"] == "replication"
    assert calls["body"]["replicatingProviderId"] == "dst-provider"
    assert calls["body"]["fileId"] == "abc123"
    assert calls["body"]["callback"] == "https://callback.example"
    assert transfer.transfer_id == "tr-repl-1"
    assert transfer.transfer_type == TransferType.REPLICATION
    assert transfer.destination_provider_id == "dst-provider"


@pytest.mark.unit
def test_replicate_alias_delegates_to_replicate_to_provider(monkeypatch, manager):
    expected = Transfer(
        transfer_id="tr-repl-alias",
        transfer_type=TransferType.REPLICATION,
        file_ref="guid#f1",
        source_provider_id=None,
        destination_provider_id="pid-dst",
    )

    def fake_replicate_to_provider(
        file_ref, destination_provider_id, callback_url=None, zone_manager=None
    ):
        assert file_ref == "guid#f1"
        assert destination_provider_id == "pid-dst"
        assert callback_url == "https://cb"
        assert zone_manager == "zone"
        return expected

    monkeypatch.setattr(manager, "replicate_to_provider", fake_replicate_to_provider)

    result = manager.replicate(
        file_ref="guid#f1",
        destination_provider_id="pid-dst",
        callback_url="https://cb",
        zone_manager="zone",
    )

    assert result is expected


@pytest.mark.unit
def test_evict_from_provider_builds_payload_and_returns_transfer(monkeypatch, manager):
    calls = {"url": None, "body": None}

    monkeypatch.setattr(manager, "resolve_provider_id", lambda provider: "resolved-src")

    def fake_raw_post(url, params=None, body=None):
        _ = params
        calls["url"] = url
        calls["body"] = body
        return {"transferId": "tr-evict-1"}

    monkeypatch.setattr(manager, "_raw_post", fake_raw_post)

    transfer = manager.evict_from_provider(
        file_ref="ch#abc123",
        source_provider="src-provider",
        callback_url="https://callback.example",
    )

    assert calls["url"].endswith("/api/v3/oneprovider/transfers")
    assert calls["body"]["type"] == "eviction"
    assert calls["body"]["evictingProviderId"] == "resolved-src"
    assert calls["body"]["fileId"] == "abc123"
    assert calls["body"]["callback"] == "https://callback.example"
    assert transfer.transfer_id == "tr-evict-1"
    assert transfer.transfer_type == TransferType.EVICTION
    assert transfer.source_provider_id == "resolved-src"


@pytest.mark.unit
def test_evict_alias_delegates_to_evict_from_provider(monkeypatch, manager):
    expected = Transfer(
        transfer_id="tr-evict-alias",
        transfer_type=TransferType.EVICTION,
        file_ref="guid#f2",
        source_provider_id="pid-src",
        destination_provider_id=None,
    )

    def fake_evict_from_provider(file_ref, source_provider, callback_url=None):
        assert file_ref == "guid#f2"
        assert source_provider == "pid-src"
        assert callback_url == "https://cb"
        return expected

    monkeypatch.setattr(manager, "evict_from_provider", fake_evict_from_provider)

    result = manager.evict(
        file_ref="guid#f2",
        source_provider_id="pid-src",
        callback_url="https://cb",
    )

    assert result is expected


@pytest.mark.unit
def test_migrate_between_providers_builds_payload_and_returns_transfer(
    monkeypatch, manager
):
    calls = {"url": None, "body": None}

    def fake_resolve_provider_id(provider):
        mapping = {
            "src-provider": "resolved-src",
            "dst-provider": "resolved-dst",
        }
        return mapping[provider]

    monkeypatch.setattr(manager, "resolve_provider_id", fake_resolve_provider_id)

    def fake_raw_post(url, params=None, body=None):
        _ = params
        calls["url"] = url
        calls["body"] = body
        return {"transferId": "tr-mig-1"}

    monkeypatch.setattr(manager, "_raw_post", fake_raw_post)

    transfer = manager.migrate_between_providers(
        file_ref="guid#xyz123",
        source_provider="src-provider",
        destination_provider="dst-provider",
        callback_url="https://callback.example",
    )

    assert calls["url"].endswith("/api/v3/oneprovider/transfers")
    assert calls["body"]["type"] == "migration"
    assert calls["body"]["evictingProviderId"] == "resolved-src"
    assert calls["body"]["replicatingProviderId"] == "resolved-dst"
    assert calls["body"]["fileId"] == "xyz123"
    assert calls["body"]["callback"] == "https://callback.example"
    assert transfer.transfer_id == "tr-mig-1"
    assert transfer.transfer_type == TransferType.MIGRATION
    assert transfer.source_provider_id == "resolved-src"
    assert transfer.destination_provider_id == "resolved-dst"


@pytest.mark.unit
def test_migrate_alias_delegates_to_migrate_between_providers(monkeypatch, manager):
    expected = Transfer(
        transfer_id="tr-mig-alias",
        transfer_type=TransferType.MIGRATION,
        file_ref="guid#f3",
        source_provider_id="pid-src",
        destination_provider_id="pid-dst",
    )

    def fake_migrate_between_providers(
        file_ref, source_provider, destination_provider, callback_url=None
    ):
        assert file_ref == "guid#f3"
        assert source_provider == "pid-src"
        assert destination_provider == "pid-dst"
        assert callback_url == "https://cb"
        return expected

    monkeypatch.setattr(
        manager, "migrate_between_providers", fake_migrate_between_providers
    )

    result = manager.migrate(
        file_ref="guid#f3",
        source_provider_id="pid-src",
        destination_provider_id="pid-dst",
        callback_url="https://cb",
    )

    assert result is expected


@pytest.mark.unit
def test_op_url_uses_guid_path_when_enabled(manager):
    manager.use_guid = True

    url = manager._op_url("guid#abc123")

    assert url.endswith("/api/v3/oneprovider/data/abc123/distribution")


@pytest.mark.unit
def test_op_url_uses_legacy_replicas_path_when_guid_disabled(manager):
    manager.use_guid = False
    manager.space_name = "my-space"

    url = manager._op_url("path/to/file")

    assert url.endswith("/api/v3/oneprovider/replicas/my-space/path/to/file")


@pytest.mark.unit
def test_raw_get_calls_session_get_and_returns_json(monkeypatch, manager):
    captured = {"url": None, "params": None, "timeout": None}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return DummyResponse(status_code=200, json_data={"ok": True})

    monkeypatch.setattr(manager._session, "get", fake_get)

    result = manager._raw_get("https://example.local/api", params={"a": 1})

    assert captured["url"] == "https://example.local/api"
    assert captured["params"] == {"a": 1}
    assert captured["timeout"] == (10, 30)
    assert result == {"ok": True}


@pytest.mark.unit
def test_get_all_file_ids_recurses_and_follows_pagination(monkeypatch, manager):
    responses = {
        "space-1": {
            "children": [
                {"file_id": "dir-1", "type": "DIR"},
                {"file_id": "file-1", "type": "REG"},
            ],
            "isLast": True,
        },
        "dir-1": {
            "children": [{"file_id": "file-2", "type": "REG"}],
            "nextPageToken": "page-2",
            "isLast": False,
        },
        "dir-1:page-2": {
            "children": [
                {"file_id": "file-2", "type": "REG"},
                {"file_id": "file-3", "type": "REG"},
            ],
            "isLast": True,
        },
    }

    def fake_raw_get(url, params=None):
        directory_id = url.rsplit("/", 2)[-2]
        key = f"{directory_id}:{params['token']}" if params.get("token") else directory_id
        return responses[key]

    monkeypatch.setattr(manager, "_raw_get", fake_raw_get)

    assert manager.get_all_file_ids() == ["file-1", "file-2", "file-3"]


@pytest.mark.unit
def test_raw_post_calls_session_post_and_returns_json(monkeypatch, manager):
    captured = {"url": None, "params": None, "json": None, "timeout": None}

    def fake_post(url, params=None, json=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse(status_code=200, json_data={"transferId": "t-1"})

    monkeypatch.setattr(manager._session, "post", fake_post)

    result = manager._raw_post(
        "https://example.local/api", params={"x": 2}, body={"k": "v"}
    )

    assert captured["url"] == "https://example.local/api"
    assert captured["params"] == {"x": 2}
    assert captured["json"] == {"k": "v"}
    assert captured["timeout"] == (10, 30)
    assert result == {"transferId": "t-1"}


@pytest.mark.unit
def test_delete_file_extracts_guid_and_calls_delete(monkeypatch, manager):
    captured = {"url": None, "timeout": None}

    def fake_delete(url, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return DummyResponse(status_code=204)

    monkeypatch.setattr(manager._session, "delete", fake_delete)

    manager.delete_file("guid#abc123")

    assert captured["url"].endswith("/api/v3/oneprovider/data/abc123")
    assert captured["timeout"] == (10, 30)


@pytest.mark.unit
def test_resolve_provider_id_matches_domain_and_alias(monkeypatch, manager):
    providers = [
        type(
            "P",
            (),
            {
                "provider_id": "pid-a",
                "domain": "site-a.provider.example.org",
                "provider_url": "https://site-a.provider.example.org",
            },
        )(),
    ]
    monkeypatch.setattr(manager, "list_providers", lambda: providers)

    assert manager.resolve_provider_id("site-a.provider.example.org") == "pid-a"
    assert manager.resolve_provider_id("site-a") == "pid-a"
    assert manager.resolve_provider_id("pid-a") == "pid-a"


@pytest.mark.unit
def test_resolve_provider_id_raises_on_unknown_provider(monkeypatch, manager):
    providers = [
        type(
            "P",
            (),
            {
                "provider_id": "pid-a",
                "domain": "site-a.provider.example.org",
                "provider_url": "https://site-a.provider.example.org",
            },
        )(),
    ]
    monkeypatch.setattr(manager, "list_providers", lambda: providers)

    with pytest.raises(ValueError):
        manager.resolve_provider_id("unknown-provider")


@pytest.mark.unit
def test_upload_file_uses_specific_space_id_and_encodes_string_content(
    monkeypatch, manager
):
    calls = {"post_url": None, "put_data": None, "put_url": None}

    def fake_post(url, timeout):
        calls["post_url"] = url
        return DummyResponse(status_code=201, json_data={"fileId": "file-123"})

    def fake_put(url, data, headers, timeout):
        calls["put_url"] = url
        calls["put_data"] = data
        assert headers["Content-Type"] == "application/octet-stream"
        return DummyResponse(status_code=204)

    monkeypatch.setattr(manager._session, "post", fake_post)
    monkeypatch.setattr(manager._session, "put", fake_put)

    file_id = manager.upload_file(
        "probe.bin", "hello", specific_space_id="devnull-space"
    )

    assert file_id == "file-123"
    assert calls["post_url"].endswith(
        "/data/devnull-space/children?name=probe.bin&type=REG"
    )
    assert calls["put_url"].endswith("/data/file-123/content")
    assert calls["put_data"] == b"hello"


@pytest.mark.unit
def test_has_file_returns_false_on_404_and_true_on_success(monkeypatch, manager):
    state = {"count": 0}

    def fake_get(url, timeout):
        _ = url
        state["count"] += 1
        if state["count"] == 1:
            return DummyResponse(status_code=404)
        return DummyResponse(status_code=200)

    monkeypatch.setattr(manager._session, "get", fake_get)

    assert manager.has_file("guid#missing") is False
    assert manager.has_file("guid#present") is True


@pytest.mark.unit
def test_create_directory_returns_existing_id_on_conflict(monkeypatch, manager):
    def fake_post(url, timeout):
        _ = (url, timeout)
        return DummyResponse(status_code=409)

    def fake_raw_get(url, params=None):
        _ = (url, params)
        return {
            "children": [
                {"name": "other", "fileId": "file-other"},
                {"name": "target", "fileId": "file-target"},
            ]
        }

    monkeypatch.setattr(manager._session, "post", fake_post)
    monkeypatch.setattr(manager, "_raw_get", fake_raw_get)

    result = manager.create_directory("parent-1", "target")
    assert result == "file-target"


@pytest.mark.unit
def test_get_transfer_status_maps_unknown_status_to_scheduled(monkeypatch, manager):
    transfer = Transfer(
        transfer_id="t-1",
        transfer_type=TransferType.REPLICATION,
        file_ref="guid#file",
        source_provider_id=None,
        destination_provider_id="pid-a",
    )

    def fake_raw_get(url, params=None):
        _ = (url, params)
        return {
            "transferStatus": "new-and-unknown",
            "bytesReplicated": 100,
            "bytesEvicted": 0,
            "filesProcessed": 1,
        }

    monkeypatch.setattr(manager, "_raw_get", fake_raw_get)

    updated = manager.update_transfer_status(transfer)
    assert updated.status == TransferStatus.SCHEDULED
    assert updated.bytes_replicated == 100
    assert updated.files_processed == 1


@pytest.mark.unit
def test_get_transfer_status_maps_replicating_to_active(monkeypatch, manager):
    transfer = Transfer(
        transfer_id="t-1b",
        transfer_type=TransferType.REPLICATION,
        file_ref="guid#file",
        source_provider_id=None,
        destination_provider_id="pid-a",
    )

    def fake_raw_get(url, params=None):
        _ = (url, params)
        return {
            "transferStatus": "replicating",
            "bytesReplicated": 100,
            "bytesEvicted": 0,
            "filesProcessed": 1,
        }

    monkeypatch.setattr(manager, "_raw_get", fake_raw_get)

    updated = manager.update_transfer_status(transfer)
    assert updated.status == TransferStatus.ACTIVE


@pytest.mark.unit
def test_get_transfer_status_overwrites_zero_timestamps(monkeypatch, manager):
    transfer = Transfer(
        transfer_id="t-1c",
        transfer_type=TransferType.REPLICATION,
        file_ref="guid#file",
        source_provider_id=None,
        destination_provider_id="pid-a",
        start_time=0,
        finish_time=0,
    )

    def fake_raw_get(url, params=None):
        _ = (url, params)
        return {
            "transferStatus": "completed",
            "startTime": 120,
            "finishTime": 145,
            "bytesReplicated": 100,
            "bytesEvicted": 0,
            "filesProcessed": 1,
        }

    monkeypatch.setattr(manager, "_raw_get", fake_raw_get)

    updated = manager.update_transfer_status(transfer)
    assert updated.start_time == 120
    assert updated.finish_time == 145
    assert updated.status == TransferStatus.COMPLETED


@pytest.mark.unit
def test_list_ended_transfer_details_returns_api_records(monkeypatch, manager):
    captured = {}

    def fake_raw_get(url, params=None):
        captured["url"] = url
        captured["params"] = params
        return {"transfers": [{"transferId": "t-1", "transferStatus": "completed"}]}

    monkeypatch.setattr(manager, "_raw_get", fake_raw_get)

    result = manager.list_ended_transfer_details(limit=25)

    assert result == [{"transferId": "t-1", "transferStatus": "completed"}]
    assert captured["url"].endswith(
        f"/api/v3/oneprovider/spaces/{manager.space_id}/transfers"
    )
    assert captured["params"] == {"state": "ended", "limit": 25}


@pytest.mark.unit
def test_wait_for_transfer_uses_byte_aware_short_circuit(monkeypatch, manager):
    transfer = Transfer(
        transfer_id="t-2",
        transfer_type=TransferType.REPLICATION,
        file_ref="guid#file",
        source_provider_id=None,
        destination_provider_id="pid-a",
    )

    snapshots = [
        {
            "transferStatus": "active",
            "bytesReplicated": 0,
            "bytesEvicted": 0,
            "filesProcessed": 0,
        },
        {
            "transferStatus": "active",
            "bytesReplicated": 2048,
            "bytesEvicted": 0,
            "filesProcessed": 1,
        },
    ]

    def fake_update_transfer_status(t):
        data = snapshots.pop(0)
        t.status = TransferStatus(data["transferStatus"])
        t.bytes_replicated = data["bytesReplicated"]
        t.bytes_evicted = data["bytesEvicted"]
        t.files_processed = data["filesProcessed"]
        return t

    monkeypatch.setattr(manager, "update_transfer_status", fake_update_transfer_status)
    monkeypatch.setattr(
        "src.rest_api.onedata.onedatamanager.time.sleep", lambda _: None
    )

    result = manager.wait_for_transfer(transfer, poll_interval=0.01, timeout=1)
    assert result.status == TransferStatus.COMPLETED


@pytest.mark.unit
def test_manager_without_provider_url_falls_back_to_onezone(monkeypatch):
    cfg = OnedataDataManager.load_config(
        token="dummy-token",
        onezone_host="onezone.example.org",
        oneprovider_host="",
        dev_mode=True,
    )
    manager = OnedataDataManager.build_storage_manager(
        config=cfg,
        provider_url=None,
        space_id="space-1",
    )

    def fake_raw_get(url, params=None):
        _ = (url, params)
        raise RuntimeError("simulate provider resolution failure")

    monkeypatch.setattr(manager, "_raw_get", fake_raw_get)

    # With no explicit provider URL and failed provider discovery,
    # implementation should fall back to Onezone proxy.
    assert manager._base_url == "https://onezone.example.org"
