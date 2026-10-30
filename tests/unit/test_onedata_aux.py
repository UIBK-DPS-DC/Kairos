import pytest

from src.rest_api.onedata.onedata_helpers import (
    ProviderSummary,
    _compute_blocks_percentage,
)


@pytest.mark.unit
def test_compute_blocks_percentage_from_distribution_blocks():
    provider_info = {
        "virtualSize": 100,
        "distributionPerStorage": {"s1": {"blocks": [[0, 30], [30, 20]]}},
    }

    result = _compute_blocks_percentage(provider_info)

    assert result == 50.0


@pytest.mark.unit
def test_compute_blocks_percentage_without_virtual_size_returns_full_when_blocks_exist():
    provider_info = {
        "virtualSize": 0,
        "distributionPerStorage": {"s1": {"blocks": [[0, 10]]}},
    }

    result = _compute_blocks_percentage(provider_info)

    assert result == 100.0


@pytest.mark.unit
def test_compute_blocks_percentage_uses_physical_size_fallback():
    provider_info = {
        "virtualSize": 200,
        "physicalSize": 50,
        "distributionPerStorage": {},
    }

    result = _compute_blocks_percentage(provider_info)

    assert result == 25.0


@pytest.mark.unit
def test_provider_url_returns_https_for_valid_domain_with_whitespace():
    provider = ProviderSummary(
        provider_id="pid-a",
        name="Provider A",
        domain=" site-b.provider.example.org   ",
    )

    assert provider.provider_url == "https://site-b.provider.example.org"


@pytest.mark.unit
def test_provider_url_returns_empty_for_invalid_domain():
    provider = ProviderSummary(
        provider_id="pid-a",
        name="Provider A",
        domain="invalid-domain",
    )

    assert provider.provider_url == ""
