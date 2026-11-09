from types import SimpleNamespace

import pytest

from src.runtime import object_access_pipeline as pipeline

NAMES = ["cloud-pl", "Cloud", "uibk", "Cloud-SK"]


@pytest.fixture
def build_managers(monkeypatch):
    class FakeManagerClass:
        @staticmethod
        def load_config(**kwargs):
            return kwargs

        @staticmethod
        def build_storage_manager(config, **kwargs):
            return SimpleNamespace(
                list_providers=lambda: [
                    SimpleNamespace(name=name, provider_id=f"id-{name}", provider_url=None)
                    for name in NAMES
                ],
                provider_name=kwargs.get("provider_name"),
            )

    monkeypatch.setattr(pipeline, "OnedataDataManager", FakeManagerClass)
    monkeypatch.setattr(
        pipeline,
        "_load_onedata_settings",
        lambda: {
            "zone_token": "zone",
            "provider_tokens": ["provider"],
            "devnull_space_id": None,
            "onezone_host": "zone-host",
            "oneprovider_host": "",
            "space_id": "space",
        },
    )

    def _build(excluded):
        if excluded is None:
            monkeypatch.delenv("EXCLUDED_PROVIDERS", raising=False)
        else:
            monkeypatch.setenv("EXCLUDED_PROVIDERS", excluded)
        return pipeline._build_onedata_managers()[1]

    return _build


@pytest.mark.parametrize("excluded", [None, "", " , "])
def test_all_providers_are_used_without_an_exclusion(build_managers, excluded):
    assert list(build_managers(excluded)) == NAMES


@pytest.mark.parametrize(
    "excluded,expected",
    [
        ("cloud-pl", ["Cloud", "uibk", "Cloud-SK"]),
        (" cloud-pl , Cloud-SK ", ["Cloud", "uibk"]),
    ],
)
def test_excluded_providers_get_no_manager(build_managers, excluded, expected):
    assert list(build_managers(excluded)) == expected


def test_excluding_every_provider_fails_instead_of_running_without_any(build_managers):
    with pytest.raises(RuntimeError, match="Could not build provider managers"):
        build_managers(",".join(NAMES))
