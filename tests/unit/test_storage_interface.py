import inspect

import pytest

from src.rest_api.onedata.onedatamanager import OnedataDataManager
from src.rest_api.storage_interface import StorageInterface


class IncompleteStorage(StorageInterface):
    """Intentionally incomplete implementation for ABC contract tests."""


class DelegatingStorage(StorageInterface):
    """Concrete subclass that delegates all methods to StorageInterface base bodies."""

    @staticmethod
    def load_config(
        token: str,
        onezone_host: str = "",
        oneprovider_host: str = "",
        dev_mode: bool = False,
    ):
        return StorageInterface.load_config(
            token=token,
            onezone_host=onezone_host,
            oneprovider_host=oneprovider_host,
            dev_mode=dev_mode,
        )

    @classmethod
    def build_storage_manager(
        cls,
        config,
        *,
        provider_url=None,
        space_name: str = "",
        space_id: str = "",
        use_guid: bool = True,
    ):
        return StorageInterface.build_storage_manager(
            config=config,
            provider_url=provider_url,
            space_name=space_name,
            space_id=space_id,
            use_guid=use_guid,
        )

    def list_providers(self) -> list:
        return super().list_providers()

    def resolve_provider_id(self, hostname_or_id: str) -> str:
        return super().resolve_provider_id(hostname_or_id)

    def update_transfer_status(self, transfer_id):
        return super().update_transfer_status(transfer_id)

    def list_ended_transfer_details(self, limit=1000):
        return super().list_ended_transfer_details(limit)

    def get_replica_distribution(self, file_ref: str):
        return super().get_replica_distribution(file_ref)

    def replicate(
        self,
        file_ref: str,
        destination_provider_id: str,
        *,
        callback_url=None,
        zone_manager=None,
    ):
        return super().replicate(
            file_ref=file_ref,
            destination_provider_id=destination_provider_id,
            callback_url=callback_url,
            zone_manager=zone_manager,
        )

    def evict(self, file_ref: str, source_provider_id: str, *, callback_url=None):
        return super().evict(
            file_ref=file_ref,
            source_provider_id=source_provider_id,
            callback_url=callback_url,
        )

    def migrate(
        self,
        file_ref: str,
        source_provider_id: str,
        destination_provider_id: str,
        *,
        callback_url=None,
    ):
        return super().migrate(
            file_ref=file_ref,
            source_provider_id=source_provider_id,
            destination_provider_id=destination_provider_id,
            callback_url=callback_url,
        )

    def wait_for_transfer(
        self, transfer, poll_interval: float = 5.0, timeout: float = 300.0
    ):
        return super().wait_for_transfer(
            transfer=transfer, poll_interval=poll_interval, timeout=timeout
        )

    def upload_file(
        self,
        filename: str,
        content: bytes | str,
        parent_id=None,
        specific_space_id=None,
    ) -> str:
        return super().upload_file(
            filename=filename,
            content=content,
            parent_id=parent_id,
            specific_space_id=specific_space_id,
        )

    def delete_file(self, file_id: str) -> None:
        return super().delete_file(file_id)

    def read_file(self, file_id: str) -> bytes:
        return super().read_file(file_id)

    def has_file(self, file_id: str) -> bool:
        return super().has_file(file_id)

    def get_file_size(self, file_id: str) -> int:
        return super().get_file_size(file_id)

    def get_data_distribution(self) -> dict:
        return super().get_data_distribution()

    def get_popularity(self, file_id: str) -> dict:
        return super().get_popularity(file_id)

    def get_file_popularity(self, zone_manager: StorageInterface) -> dict:
        return super().get_file_popularity(zone_manager)


def _build_onedata_manager() -> StorageInterface:
    """Build an Onedata manager instance without doing live network calls."""
    cfg = OnedataDataManager.load_config(
        token="dummy-token",
        onezone_host="onezone.example.org",
        oneprovider_host="provider.example.org",
        dev_mode=True,
    )
    return OnedataDataManager.build_storage_manager(
        config=cfg,
        provider_url="https://provider.example.org",
        space_id="dummy-space",
    )


# Add future backends here as they are implemented.
IMPLEMENTATIONS = [
    pytest.param(
        {
            "name": "onedata",
            "class": OnedataDataManager,
            "factory": _build_onedata_manager,
        },
        id="onedata",
    )
]


@pytest.mark.unit
def test_storage_interface_is_abstract():
    """The base interface cannot be instantiated directly."""
    with pytest.raises(TypeError):
        StorageInterface()


@pytest.mark.unit
def test_incomplete_storage_implementation_is_rejected():
    """ABC enforcement should reject subclasses missing required methods."""
    with pytest.raises(TypeError):
        IncompleteStorage()


@pytest.mark.unit
def test_storage_interface_base_methods_raise_not_implemented_error():
    """All base method stubs raise NotImplementedError when directly delegated to."""
    storage = DelegatingStorage()

    checks = [
        lambda: DelegatingStorage.load_config(token="t"),
        lambda: DelegatingStorage.build_storage_manager(config={}),
        lambda: storage.list_providers(),
        lambda: storage.resolve_provider_id("provider"),
        lambda: storage.get_replica_distribution("file"),
        lambda: storage.replicate("file", "dst"),
        lambda: storage.evict("file", "src"),
        lambda: storage.migrate("file", "src", "dst"),
        lambda: storage.wait_for_transfer(transfer={}),
        lambda: storage.upload_file(filename="x", content=b"y"),
        lambda: storage.delete_file("id"),
        lambda: storage.read_file("id"),
        lambda: storage.has_file("id"),
        lambda: storage.get_file_size("id"),
        lambda: storage.get_file_popularity(zone_manager=storage),
        lambda: storage.get_data_distribution(),
        lambda: storage.update_transfer_status("transfer_id"),
        lambda: storage.list_ended_transfer_details(),
    ]

    for call in checks:
        with pytest.raises(NotImplementedError):
            call()


@pytest.mark.unit
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_registered_implementation_is_concrete_storage_interface(impl):
    """Registered backend classes must be concrete StorageInterface implementations."""
    impl_cls = impl["class"]

    assert issubclass(impl_cls, StorageInterface)
    assert not inspect.isabstract(impl_cls)


@pytest.mark.unit
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_registered_implementation_factory_returns_interface_instance(impl):
    """Each backend factory should return a StorageInterface instance."""
    storage = impl["factory"]()

    assert isinstance(storage, StorageInterface)


@pytest.mark.unit
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_upload_file_signature_supports_expected_keyword_args(impl):
    """Upload contract should support parent and space-specific routing kwargs."""
    signature = inspect.signature(impl["class"].upload_file)
    params = signature.parameters

    assert "filename" in params
    assert "content" in params
    assert "parent_id" in params
    assert "specific_space_id" in params


@pytest.mark.unit
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_build_storage_manager_signature_is_interface_compatible(impl):
    """Backend builder should expose core interface-level constructor options."""
    signature = inspect.signature(impl["class"].build_storage_manager)
    params = signature.parameters

    assert "config" in params
    assert "provider_url" in params
    assert "space_name" in params
    assert "space_id" in params
    assert "use_guid" in params
