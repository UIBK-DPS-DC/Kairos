"""High-level Onedata data manager.

This module contains runtime API logic for provider discovery, transfer
operations, and file lifecycle methods.

Shared models, config helpers, and supporting transfer/data classes are
defined in `onedata_helpers.py`.
"""

import logging
import time
import requests
from typing import Optional

from src.rest_api.onedata.onedata_helpers import (
    OnedataConfig,
    ProviderSummary,
    ReplicaDistribution,
    Transfer,
    TransferStatus,
    TransferType,
    load_config as aux_load_config,
    _compute_blocks_percentage,
)
from src.rest_api.storage_interface import StorageInterface

logger = logging.getLogger(__name__)


class OnedataDataManager(StorageInterface):
    """
    Data lifecycle manager for Onedata files.

    All file_ref arguments accept GUID strings in any of the three formats
    that the Prometheus monitor may emit:  "guid#<hex>", "ch#<hex>", or plain hex.

    Parameters
    ----------
    config : OnedataConfig
        Auth token, Onezone URL, and optional Oneprovider URL.
    provider_url : str, optional
        Explicit Oneprovider base URL, e.g. "https://provider.example.org".
        Overrides config.oneprovider_url. If omitted, the first reachable provider
        is resolved lazily from Onezone.
    space_name : str
        Only needed when use_guid=False (legacy path-based replicas endpoint).
    use_guid : bool
        True  (default) → /data/{file_id}/distribution endpoints (modern API).
        False            → /replicas/{space}/{path} legacy endpoints (deprecated).
    """

    PROVIDER_BASE = "/api/v3/oneprovider"
    ZONE_BASE = "/api/v3/onezone"

    @staticmethod
    def load_config(
        token: str,
        onezone_host: str = "",
        oneprovider_host: str = "",
        dev_mode: bool = False,
    ) -> OnedataConfig:
        """Create an OnedataConfig from token and host values."""
        return aux_load_config(
            token=token,
            onezone_host=onezone_host,
            oneprovider_host=oneprovider_host,
            dev_mode=dev_mode,
        )

    @classmethod
    def build_storage_manager(
        cls,
        config: OnedataConfig,
        *,
        provider_url: Optional[str] = None,
        provider_name: str = "",
        space_name: str = "",
        space_id: str = "",
        use_guid: bool = True,
    ) -> StorageInterface:
        """Create the default Onedata-backed StorageInterface implementation."""
        return cls(
            config=config,
            provider_url=provider_url,
            provider_name=provider_name,
            space_name=space_name,
            space_id=space_id,
            use_guid=use_guid,
        )

    def __init__(
        self,
        config: OnedataConfig,
        provider_url: Optional[str] = None,
        provider_name: str = "",
        space_name: str = "",
        space_id: str = "",
        use_guid: bool = True,
    ):
        self.config = config
        self.provider_name = provider_name
        self.space_name = space_name
        self.space_id = space_id
        self.use_guid = use_guid
        self._session = self._build_session()

        if provider_url:
            self.__provider_url: Optional[str] = provider_url
        elif config.oneprovider_url:
            self.__provider_url = config.oneprovider_url
        else:
            self.__provider_url = None  # resolved lazily

    # ------------------------------------------------------------------
    # Base URL resolution
    # ------------------------------------------------------------------

    @property
    def _base_url(self) -> str:
        """
        Return the Oneprovider base URL, resolving from Onezone if needed.

        Resolution order
        ----------------
        1. Explicit ``provider_url`` passed to __init__ or set in config — used as-is.
        2. First accessible provider whose ``domain`` field is a non-empty, resolvable
           hostname (fetched from Onezone).
        3. Onezone itself as a transparent proxy — Onezone forwards Oneprovider API
           calls to a suitable provider automatically.  This is the fallback used by
           the official EGI DataHub documentation and works whenever the token grants
           access to at least one space.

        Raises RuntimeError only when Onezone is also unreachable.
        """
        if self.__provider_url:
            return self.__provider_url

        logger.debug("ONEPROVIDER_HOST not set — resolving provider URL from Onezone")

        # Try to pick a real Oneprovider domain from the accessible providers list.
        try:
            pids_url = (
                f"{self.config.onezone_url}{self.ZONE_BASE}/user/effective_providers"
            )
            data = self._raw_get(pids_url)
            for pid in data.get("providers", []):
                pdata = self._raw_get(
                    f"{self.config.onezone_url}{self.ZONE_BASE}/providers/{pid}"
                )
                domain = pdata.get("domain", "").strip()
                # A valid domain contains at least one dot and no spaces.
                # Opaque provider IDs (hex strings ending in "ch…") are not domains.
                if (
                    domain
                    and "." in domain
                    and " " not in domain
                    and not domain.endswith("ch")
                ):
                    self.__provider_url = f"https://{domain}"
                    logger.info(
                        "Auto-resolved Oneprovider URL: %s", self.__provider_url
                    )
                    return self.__provider_url
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Provider domain resolution failed (%s); falling back to Onezone proxy",
                exc,
            )

        # Fallback: Onezone proxies Oneprovider API requests transparently.
        # This is the pattern used in official EGI DataHub examples:
        #   curl -H "X-Auth-Token: $TOKEN" https://$ONEZONE_HOST/api/v3/oneprovider/…
        if self.config.onezone_url:
            logger.info(
                "No resolvable Oneprovider domain found; using Onezone as API proxy: %s",
                self.config.onezone_url,
            )
            self.__provider_url = self.config.onezone_url
            return self.__provider_url

        raise RuntimeError(
            "Cannot determine a Oneprovider URL: no provider_url configured, "
            "no resolvable provider domain found via Onezone, and onezone_url is empty.\n"
            "Set ONEDATA_PROVIDER_HOST or pass provider_url= to OnedataDataManager()."
        )

    @property
    def _provider_api_base(self) -> str:
        """
        Base URL to use for ALL Oneprovider API calls (/api/v3/oneprovider/…).

        We bypass the Onezone proxy and hit the Oneprovider directly because
        Onezone does not support proxying the /transfers endpoints (returns 405).
        """
        return self._base_url

    # ------------------------------------------------------------------
    # GUID normaliser
    # ------------------------------------------------------------------

    @staticmethod
    def extract_guid(raw: str) -> str:
        """
        Strip monitor-added prefixes from file_id values.
            "guid#<hex>"  →  "<hex>"
            "ch#<hex>"    →  "<hex>"
            "<plain>"     →  "<plain>"
        """
        if not raw or not isinstance(raw, str):
            return raw
        if "guid#" in raw:
            return raw.split("guid#", 1)[1]
        if raw.startswith("ch#"):
            return raw[3:]
        return raw

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_providers(self) -> list:
        """
        Return all Oneproviders reachable with this token as ProviderSummary objects.
        Use the provider_id values for replicate() / evict() / migrate().
        GET https://<onezone-host>/api/v3/onezone/user/spaces/<space-id>
        """
        url = f"{self.config.onezone_url}{self.ZONE_BASE}/user/spaces/{self.space_id}"
        data = self._raw_get(url)
        providers = []
        for pid in data.get("providers", {}).keys():
            pdata = self._raw_get(
                f"{self.config.onezone_url}{self.ZONE_BASE}/providers/{pid}"
            )
            name = pdata.get("name", pid)
            domain = pdata.get("domain", "").strip()

            providers.append(
                ProviderSummary(
                    provider_id=pid,
                    name=name,
                    domain=domain,
                )
            )
        logger.debug("Providers: %s", providers)
        return providers

    def resolve_provider_id(self, hostname_or_id: str) -> str:
        """
        Accept either an opaque provider ID or a hostname/domain and always
        return an opaque provider ID.

        Matching order
        --------------
        1. Exact provider ID match.
        2. Domain / FQDN match (case-insensitive, ignoring trailing slashes).
        3. Short domain alias match (first label before the first dot).

        Raises
        ------
        ValueError  if no provider can be matched.
        """
        providers = self.list_providers()

        needle = (
            hostname_or_id.lower()
            .removeprefix("https://")
            .removeprefix("http://")
            .rstrip("/")
        )

        # 1. Exact ID match
        for p in providers:
            if p.provider_id == hostname_or_id or p.provider_id.lower() == needle:
                return p.provider_id

        # 2. Domain/FQDN match
        for p in providers:
            if p.domain.lower() == needle:
                return p.provider_id

        # 3. Short alias match from domain prefix, e.g. "site-a" for
        # "site-a.provider.example.org".
        for p in providers:
            domain = p.domain.lower()
            if not domain:
                continue
            short_alias = domain.split(".", 1)[0]
            if short_alias == needle:
                return p.provider_id

        known = "\n  ".join(
            f"{p.provider_id}  domain={p.domain!r}  url={p.provider_url!r}"
            for p in providers
        )
        raise ValueError(
            f"Cannot resolve {hostname_or_id!r} to a provider ID.\n"
            f"Accessible providers:\n  {known}"
        )

    def infer_token_scope(self) -> dict:
        """
        Call POST /onezone/tokens/infer_access_token_scope and return the raw
        JSON.  Useful for diagnosing 401 errors: verifies that the token has
        the ``rest`` interface caveat and is not expired.

        Returns the raw response dict from Onezone.
        """
        url = (
            f"{self.config.onezone_url}{self.ZONE_BASE}/tokens/infer_access_token_scope"
        )
        r = self._session.post(
            url,
            json={"token": self.config.token},
            timeout=(10, 30),
        )
        r.raise_for_status()
        return r.json()

    def diagnose_token(self) -> dict:
        """
        Return a structured diagnosis of the current token's caveats.

        Returns a dict with keys:
            valid          bool   — Onezone accepted the token
            caveats        list   — raw caveat list from Onezone
            interfaces     list   — values of all interface caveats
            has_rest       bool   — True if interface=rest caveat present
            service_ids    list   — values of any service caveats (opaque IDs)
                                    Non-empty means the token is locked to specific
                                    services and will 401 on providers not listed.
            ip_ranges      list   — IP range caveats if any
            error          str    — error message if the call failed
        """
        result = {
            "valid": False,
            "caveats": [],
            "interfaces": [],
            "has_rest": False,
            "service_ids": [],
            "ip_ranges": [],
            "error": "",
        }
        try:
            scope = self.infer_token_scope()
            result["valid"] = True
            caveats = scope.get("tokenCaveats", scope.get("caveats", []))
            result["caveats"] = caveats
            for c in caveats:
                t = c.get("type", "")
                if t == "interface":
                    result["interfaces"].append(c.get("interface", ""))
                elif t == "service":
                    result["service_ids"].append(c.get("service", str(c)))
                elif t == "ip":
                    result["ip_ranges"].append(c.get("whitelist", c.get("ip", str(c))))
            result["has_rest"] = "rest" in result["interfaces"]
        except Exception as exc:  # noqa: BLE001
            result["error"] = str(exc)
        return result

    def has_rest_caveat(self) -> bool:
        """
        Return True if the current token is known to have the ``interface: rest``
        caveat.

        NOTE: On some Onezone deployments (e.g. EGI DataHub)
        ``infer_access_token_scope`` only returns ``dataAccessScope`` and does NOT
        expose the token's own caveats.  In that case this method always returns
        False, which causes ``ensure_rest_token`` to mint a new token.  That is
        the desired behaviour — the minted token will have the correct caveats.
        """
        try:
            scope = self.infer_token_scope()
        except Exception:
            return False
        # Some versions return caveats here; most return only dataAccessScope.
        caveats = scope.get("tokenCaveats", scope.get("caveats", []))
        interfaces = [
            c.get("interface")
            for c in caveats
            if isinstance(c, dict) and c.get("type") == "interface"
        ]
        return "rest" in interfaces

    def mint_rest_token(self, name: str = "onedata-manager-rest-token") -> str:
        """
        Create (or replace) a named Onezone access token that works on all
        accessible Oneproviders, and return the raw token string.

        The minted token has:
          - ``interface: rest``   — required for REST API calls
          - ``service`` whitelist — includes ``ozw-onezone`` (so the token can be
            used via the Onezone proxy) plus ``opw-<id>`` for every provider the
            current token can access (so direct provider calls also work)

        If a token with the same name already exists it is deleted first (by ID,
        since this Onezone version does not support delete-by-name).
        """
        zone = self.config.onezone_url
        list_url = f"{zone}{self.ZONE_BASE}/user/tokens/named"

        # ── 1. Delete any existing token with this name (look up its ID first) ──
        try:
            existing = self._raw_get(list_url)
            for tid in existing.get("tokens", []):
                try:
                    tdata = self._raw_get(f"{list_url}/{tid}")
                    if tdata.get("name") == name:
                        r = self._session.delete(f"{list_url}/{tid}", timeout=(10, 30))
                        if r.status_code not in (200, 204, 404):
                            r.raise_for_status()
                        logger.debug("Deleted existing token %r (id=%s)", name, tid)
                        break
                except Exception as _e:
                    logger.debug("Could not inspect/delete token %s: %s", tid, _e)
        except Exception as exc:
            logger.debug("Could not list tokens for cleanup: %s", exc)

        # ── 2. Build service-caveat whitelist ────────────────────────────────────
        # Always include Onezone so the token works via the Onezone proxy.
        # Also include every accessible provider so direct calls work too.
        service_whitelist = ["ozw-onezone"]
        try:
            for p in self.list_providers():
                # Provider service IDs are "opw-<provider_id>"
                service_whitelist.append(f"opw-{p.provider_id}")
        except Exception as exc:
            logger.debug("Could not list providers for service caveat: %s", exc)

        body = {
            "name": name,
            "type": {"accessToken": {}},
            "caveats": [
                {"type": "interface", "interface": "rest"},
                {"type": "service", "whitelist": service_whitelist},
            ],
        }
        r = self._session.post(list_url, json=body, timeout=(10, 30))

        if not r.ok:
            try:
                err_body = r.json()
                description = (
                    err_body.get("description")
                    or err_body.get("error", {}).get("description")
                    or str(err_body)
                )
            except Exception:
                description = r.text or "(empty response body)"

            if r.status_code == 400:
                raise RuntimeError(
                    f"Onezone rejected the token-creation request (400 Bad Request).\n"
                    f"Server said: {description}\n\n"
                    f"Most common causes:\n"
                    f"  1. The current token is an OIDC/federated token (egi:…) that\n"
                    f"     cannot create child tokens — use a native Onezone token.\n"
                    f"  2. The token name is already taken and delete-by-ID failed.\n"
                    f"     Go to Onezone UI → Access Tokens, delete '{name}', retry.\n\n"
                    f"Workaround: create the token manually in the Onezone UI:\n"
                    f"  Access Tokens → New token\n"
                    f"  Add caveat: Interface = REST\n"
                    f"  Add caveat: Service = all providers + onezone\n"
                    f"then set ONEDATA_TOKEN=<new token> and re-run."
                )
            r.raise_for_status()

        data = r.json()
        token_str = data.get("token")
        if not token_str:
            raise RuntimeError(
                f"Onezone returned no token string.  Full response: {data}"
            )
        logger.info(
            "Minted REST token: name=%r  services=%s  token=%.20s…",
            name,
            service_whitelist,
            token_str,
        )
        return token_str

    def ensure_rest_token(self, name: str = "onedata-manager-rest-token") -> bool:
        """
        Mint a properly-scoped REST token and update the session in-place.

        On deployments like EGI DataHub, tokens have a hidden
        ``service`` caveat (``{"whitelist": ["ozw-onezone"]}``) that is not
        visible via ``infer_access_token_scope``.  This method always mints a
        new token with both ``interface: rest`` and a ``service`` whitelist
        covering all accessible providers, so that direct provider calls work.

        Returns
        -------
        bool  — always True (token was replaced).
        """
        new_token = self.mint_rest_token(name=name)
        self.config = OnedataConfig(
            token=new_token,
            onezone_url=self.config.onezone_url,
            oneprovider_url=self.config.oneprovider_url,
            verify_ssl=self.config.verify_ssl,
        )
        self._session = self._build_session()
        logger.info("Session updated with new REST+service-scoped token.")
        return True

    # ------------------------------------------------------------------
    # Distribution query
    # ------------------------------------------------------------------

    def get_replica_distribution(self, file_ref: str) -> ReplicaDistribution:
        """
        Return per-provider replica block coverage for a file.

        Uses GET /data/{id}/distribution (modern stable API).
        The response key is "distributionPerProvider"; each value contains
        "blocksPercentage" (0–100) and "virtualSize" (bytes).

        Returns
        -------
        ReplicaDistribution
            .provider_ids              — providers holding any blocks (percentage > 0)
            .fully_replicated_providers— providers with blocksPercentage == 100
            .replica_count             — number of providers holding any blocks
            .blocks_percentage(pid)    — coverage % for a specific provider
            .virtual_size(pid)         — virtual file size at a specific provider
        """
        url = f"{self._provider_api_base}{self.PROVIDER_BASE}/data/{file_ref}/distribution"
        data = self._raw_get(url)
        dist = ReplicaDistribution(file_ref=file_ref)

        if self.use_guid:
            # Modern stable API: distributionPerProvider
            # Each entry has: blocksPercentage, virtualSize, chunksBarWidth,
            # distributionPerStorage (nested).
            # Some deployments omit blocksPercentage; we compute it ourselves
            # from distributionPerStorage block ranges when it is missing.
            for pid, info in data.get("distributionPerProvider", {}).items():
                if "blocksPercentage" not in info:
                    info = dict(info)  # don't mutate the original
                    info["blocksPercentage"] = _compute_blocks_percentage(info)
                dist.distribution_per_provider[pid] = info
        else:
            # Legacy /replicas/ endpoint (deprecated).
            # Returns replicasPerProvider: { pid: [[offset, size], ...] }
            # Approximate "complete" as any blocks present; no percentage available.
            for pid, blocks in data.get("replicasPerProvider", {}).items():
                dist.distribution_per_provider[pid] = {
                    "blocksPercentage": 100.0 if blocks else 0.0,
                    "virtualSize": 0,
                    "_legacyBlocks": blocks,
                }

        return dist

    def get_data_distribution_of_provider(self) -> dict:
        """
        Return raw distribution data for the provider.

        Returns
        -------
        dict
            Raw JSON payload returned by the distribution endpoint.
        """

        url = f"{self._provider_api_base}{self.PROVIDER_BASE}/data/{self.space_id}/distribution"
        return self._raw_get(url)

    def get_data_distribution(self) -> dict:
        """
        Return provider-level physical sizes as {provider_id: physical_size}.

        The source payload stores sizes under:
          distributionPerProvider -> <provider_id> -> distributionPerStorageBackend
        If a provider has multiple storage backends, their physical sizes are summed.
        """
        response_json = self.get_data_distribution_of_provider()
        distribution = response_json.get("distributionPerProvider", {})

        provider_physical_sizes: dict[str, int] = {}
        for provider_id, provider_info in distribution.items():
            storage_backends = provider_info.get("distributionPerStorageBackend", {})
            total_physical_size = 0

            for backend_info in storage_backends.values():
                physical_size = backend_info.get("physicalSize", 0)
                if isinstance(physical_size, (int, float)):
                    total_physical_size += int(physical_size)

            provider_physical_sizes[provider_id] = total_physical_size

        return provider_physical_sizes

    # ------------------------------------------------------------------
    # Core transfer operations
    # ------------------------------------------------------------------

    def replicate_to_provider(
        self,
        file_ref: str,
        destination_provider_id: str,
        *,
        callback_url: Optional[str] = None,
        zone_manager: Optional[StorageInterface] = None,
    ) -> Transfer:
        """
        Start a replication transfer — copy the file to provider with destination_provider_id.

        Parameters
        ----------
        file_ref : str
            File GUID (any accepted format).
        destination_provider_id : str
            Target Oneprovider provider ID.
        callback_url : str, optional
            Onedata will POST to this URL when the transfer finishes.
        zone_manager : StorageInterface, optional
            Accepted for backward compatibility and ignored.

        Returns
        -------
        Transfer  (status=SCHEDULED; poll with get_transfer_status / wait_for_transfer)
        """

        body: dict = {
            "type": "replication",
            "replicatingProviderId": destination_provider_id,
            "dataSourceType": "file",
            "fileId": self.extract_guid(file_ref),
        }
        if callback_url:
            body["callback"] = callback_url

        # Kept for interface/backward compatibility; this method already receives
        # destination provider IDs directly.
        _ = zone_manager

        data = self._raw_post(
            f"{self._provider_api_base}{self.PROVIDER_BASE}/transfers", body=body
        )
        tid = data["transferId"]
        logger.info(
            "Replication started → %s  file=%.20s  transfer=%s",
            destination_provider_id,
            file_ref,
            tid,
        )
        return Transfer(
            transfer_id=tid,
            transfer_type=TransferType.REPLICATION,
            file_ref=file_ref,
            source_provider_id=None,
            destination_provider_id=destination_provider_id,
        )

    def replicate(
        self,
        file_ref: str,
        destination_provider_id: str,
        *,
        callback_url: Optional[str] = None,
        zone_manager: Optional[StorageInterface] = None,
    ) -> Transfer:
        """Backward-compatible alias accepting provider IDs."""
        return self.replicate_to_provider(
            file_ref=file_ref,
            destination_provider_id=destination_provider_id,
            callback_url=callback_url,
            zone_manager=zone_manager,
        )

    def evict_from_provider(
        self,
        file_ref: str,
        source_provider: str,
        *,
        callback_url: Optional[str] = None,
    ) -> Transfer:
        """
        Start an eviction transfer — remove the local replica from source_provider_id.

        The file must exist on at least one other provider before eviction is
        permitted by Onedata.

        Parameters
        ----------
        file_ref : str
            File GUID (any accepted format).
        source_provider : str
            Oneprovider host/domain or opaque provider ID whose replica should be removed.
        callback_url : str, optional
            Onedata will POST to this URL when the transfer finishes.

        Returns
        -------
        Transfer
        """
        source_provider_id = self.resolve_provider_id(source_provider)

        body: dict = {
            "type": "eviction",
            "evictingProviderId": source_provider_id,
            "dataSourceType": "file",
            "fileId": self.extract_guid(file_ref),
        }
        if callback_url:
            body["callback"] = callback_url

        data = self._raw_post(
            f"{self._provider_api_base}{self.PROVIDER_BASE}/transfers", body=body
        )
        tid = data["transferId"]
        logger.info(
            "Eviction started  src=%s  file=%.20s  transfer=%s",
            source_provider_id,
            file_ref,
            tid,
        )
        return Transfer(
            transfer_id=tid,
            transfer_type=TransferType.EVICTION,
            file_ref=file_ref,
            source_provider_id=source_provider_id,
            destination_provider_id=None,
        )

    def evict_from_provider_id(
        self,
        file_ref: str,
        source_provider_id: str,
        *,
        callback_url: Optional[str] = None,
    ) -> Transfer:
        """
        Start an eviction transfer — remove the local replica from source_provider_id.

        The file must exist on at least one other provider before eviction is
        permitted by Onedata.

        Parameters
        ----------
        file_ref : str
            File GUID (any accepted format).
        source_provider_id : str
            Opaque provider ID whose replica should be removed.
        callback_url : str, optional
            Onedata will POST to this URL when the transfer finishes.

        Returns
        -------
        Transfer
        """
        
        body: dict = {
            "type": "eviction",
            "evictingProviderId": source_provider_id,
            "dataSourceType": "file",
            "fileId": self.extract_guid(file_ref),
        }
        if callback_url:
            body["callback"] = callback_url

        data = self._raw_post(
            f"{self._provider_api_base}{self.PROVIDER_BASE}/transfers", body=body
        )
        tid = data["transferId"]
        logger.info(
            "Eviction started  src=%s  file=%.20s  transfer=%s",
            source_provider_id,
            file_ref,
            tid,
        )
        return Transfer(
            transfer_id=tid,
            transfer_type=TransferType.EVICTION,
            file_ref=file_ref,
            source_provider_id=source_provider_id,
            destination_provider_id=None,
        )

    def evict(
        self,
        file_ref: str,
        source_provider_id: str,
        *,
        callback_url: Optional[str] = None,
    ) -> Transfer:
        """Backward-compatible alias accepting provider IDs."""
        return self.evict_from_provider(
            file_ref=file_ref,
            source_provider=source_provider_id,
            callback_url=callback_url,
        )

    def migrate_between_providers(
        self,
        file_ref: str,
        source_provider: str,
        destination_provider: str,
        *,
        callback_url: Optional[str] = None,
    ) -> Transfer:
        """
        Start a migration transfer — atomic replicate + evict.

        Onedata copies the file to destination_provider_id and only evicts it
        from source_provider_id once the copy is confirmed complete.

        Parameters
        ----------
        file_ref : str
            File GUID (any accepted format).
        source_provider : str
            Provider host/domain or opaque ID whose replica will be evicted.
        destination_provider : str
            Target provider host/domain or opaque ID that will receive the replica.
        callback_url : str, optional
            Onedata will POST to this URL when the transfer finishes.

        Returns
        -------
        Transfer
        """
        source_provider_id = self.resolve_provider_id(source_provider)
        destination_provider_id = self.resolve_provider_id(destination_provider)

        body: dict = {
            "type": "migration",
            "replicatingProviderId": destination_provider_id,
            "evictingProviderId": source_provider_id,
            "dataSourceType": "file",
            "fileId": self.extract_guid(file_ref),
        }
        if callback_url:
            body["callback"] = callback_url

        data = self._raw_post(
            f"{self._provider_api_base}{self.PROVIDER_BASE}/transfers", body=body
        )
        tid = data["transferId"]
        logger.info(
            "Migration started  %s → %s  file=%.20s  transfer=%s",
            source_provider_id,
            destination_provider_id,
            file_ref,
            tid,
        )
        return Transfer(
            transfer_id=tid,
            transfer_type=TransferType.MIGRATION,
            file_ref=file_ref,
            source_provider_id=source_provider_id,
            destination_provider_id=destination_provider_id,
        )

    def migrate(
        self,
        file_ref: str,
        source_provider_id: str,
        destination_provider_id: str,
        *,
        callback_url: Optional[str] = None,
    ) -> Transfer:
        """Backward-compatible alias accepting provider IDs."""
        return self.migrate_between_providers(
            file_ref=file_ref,
            source_provider=source_provider_id,
            destination_provider=destination_provider_id,
            callback_url=callback_url,
        )

    # ------------------------------------------------------------------
    # Transfer lifecycle
    # ------------------------------------------------------------------

    def update_transfer_status(self, transfer: Transfer) -> Transfer:
        """
        Fetch current status from GET /transfers/{tid} and update the Transfer in-place.

        The stable API returns the status under the "status" key.
        bytesReplicated and bytesEvicted are tracked separately, as both can
        be populated for migration transfers.

        Transfer status is always polled via the Onezone proxy regardless of
        which provider URL is configured.  This is necessary because the transfer
        is tracked by the provider that owns the data, which may differ from
        the provider in ONEDATA_PROVIDER_HOST.  Onezone routes the status
        request to the correct provider automatically.

        Returns
        -------
        Transfer  (same object, mutated)
        """
        # Route via _provider_api_base (always Onezone proxy unless an explicit
        # provider URL is set) — the transfer lives on whichever provider owns
        # the data and Onezone routes the status request correctly regardless.
        base = self._provider_api_base
        url = f"{base}{self.PROVIDER_BASE}/transfers/{transfer.transfer_id}"
        data = self._raw_get(url)

        raw_status = data.get("transferStatus", "scheduled")  # default to scheduled

        # Some API responses return 0 or "0" before timestamps are finalized.
        # Treat non-positive values as missing so later valid values can overwrite.
        def _parse_timestamp(value):
            if value is None:
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if parsed > 0 else None

        start_time = _parse_timestamp(data.get("startTime"))
        if start_time is not None and (
            transfer.start_time is None or transfer.start_time <= 0
        ):
            transfer.start_time = start_time
        finish_time = _parse_timestamp(data.get("finishTime"))
        if finish_time is not None and (
            transfer.finish_time is None or transfer.finish_time <= 0
        ):
            transfer.finish_time = finish_time

        status_aliases = {
            "replicating": TransferStatus.ACTIVE,
            "evicting": TransferStatus.ACTIVE,
            "migrating": TransferStatus.ACTIVE,
        }
        normalized_status = status_aliases.get(str(raw_status).lower(), raw_status)

        try:
            transfer.status = TransferStatus(normalized_status)
        except ValueError:
            logger.warning(
                "Unknown transfer status %r — treating as SCHEDULED", raw_status
            )
            transfer.status = TransferStatus.SCHEDULED

        # Track replication and eviction byte counts separately (both populated
        # for migration transfers).
        transfer.bytes_replicated = data.get("bytesReplicated", 0) or 0
        transfer.bytes_evicted = data.get("bytesEvicted", 0) or 0
        transfer.files_processed = data.get("filesProcessed", 0) or 0
        return transfer

    def get_transfer_status(self, transfer_id: str) -> dict:
        """Return the latest raw transfer status by transfer ID."""
        if not transfer_id:
            raise ValueError("transfer_id is required")

        url = (
            f"{self._provider_api_base}"
            f"{self.PROVIDER_BASE}/transfers/{transfer_id}"
        )
        return self._raw_get(url)

    def list_ended_transfer_details(self, limit: int = 1000) -> list[dict]:
        """Return transfer detail records for this manager's space.

        The correct Oneprovider endpoint is per-space:
            GET /api/v3/oneprovider/spaces/{space_id}/transfers

        The raw records are preserved because deployments expose slightly
        different detail fields; interval metrics can normalize the fields
        they need without losing provider-specific information.
        """
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        data = self._raw_get(
            f"{self._provider_api_base}{self.PROVIDER_BASE}/spaces/{self.space_id}/transfers",
            params={"state": "ended", "limit": limit},
        )
        if isinstance(data, list):
            return data
        transfers = data.get("transfers", [])
        if not isinstance(transfers, list):
            raise RuntimeError(
                "Unexpected transfer-list response: 'transfers' is not a list"
            )
        return transfers

    def wait_for_transfer(
        self,
        transfer: Transfer,
        poll_interval: float = 5.0,
        timeout: float = 300.0,
    ) -> Transfer:
        """
        Block until the transfer reaches a terminal state, logging progress.
        Includes a byte-aware short-circuit for lagging API statuses.
        """
        self.update_transfer_status(transfer)
        logger.info(
            "Waiting for transfer %s [%s]  initial status: %s",
            transfer.transfer_id,
            transfer.transfer_type.value,
            transfer.status.value,
        )

        elapsed = 0.0
        try:
            while (
                not transfer.is_terminal
                and not transfer.status == TransferStatus.COMPLETED
            ):
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Transfer {transfer.transfer_id} did not finish within {timeout}s "
                        f"(last status: {transfer.status}). "
                    )
                time.sleep(poll_interval)
                elapsed += poll_interval
                self.update_transfer_status(transfer)

                logger.info(
                    "Transfer %s: %-10s  replicated=%d B  evicted=%d B  "
                    "files=%d  (%.0fs elapsed)",
                    transfer.transfer_id,
                    transfer.status.value,
                    transfer.bytes_replicated,
                    transfer.bytes_evicted,
                    transfer.files_processed,
                    elapsed,
                )

                # I think this is better because it comes back from the response
                if transfer.status == TransferStatus.COMPLETED:
                    break

                # --- BYTE-AWARE SHORT-CIRCUIT ---
                # Onedata's job status often lags behind the actual physical block operations.
                # We verify completion by checking that the bytes have actually moved/dropped.
                is_done = False
                if transfer.files_processed >= 1:
                    if (
                        transfer.transfer_type == TransferType.REPLICATION
                        and transfer.bytes_replicated > 0
                    ):
                        is_done = True
                    elif (
                        transfer.transfer_type == TransferType.EVICTION
                        and transfer.bytes_evicted > 0
                    ):
                        is_done = True
                    elif (
                        transfer.transfer_type == TransferType.MIGRATION
                        and transfer.bytes_replicated > 0
                        and transfer.bytes_evicted > 0
                    ):
                        is_done = True

                if is_done:
                    logger.info(
                        "  -> Physical byte transfer confirmed. Bypassing lagging API status."
                    )
                    transfer.status = TransferStatus.COMPLETED
                    break

        except KeyboardInterrupt:
            logger.warning("wait_for_transfer interrupted.")
            raise

        if transfer.status == TransferStatus.FAILED:
            raise RuntimeError(f"Transfer {transfer.transfer_id} failed.")

        return transfer

    # ------------------------------------------------------------------
    # File creation / upload
    # ------------------------------------------------------------------

    def create_directory(self, parent_id: str, name: str) -> str:
        """
        Create a directory under *parent_id* and return its file ID.
        Idempotent: on HTTP 409 (already exists) the existing ID is returned.

        Parameters
        ----------
        parent_id : str
            GUID of the parent directory (or space root ID).
        name : str
            Directory name to create.
        """
        base = self._base_url
        url = (
            f"{base}{self.PROVIDER_BASE}/data/{parent_id}/children?name={name}&type=DIR"
        )
        r = self._session.post(url, timeout=(10, 30))
        if r.status_code in (201, 200):
            return r.json()["fileId"]
        if r.status_code == 409:
            # Already exists — find it by listing children
            list_url = f"{base}{self.PROVIDER_BASE}/data/{parent_id}/children"
            data = self._raw_get(list_url, params={"limit": 1000})
            for child in data.get("children", []):
                if child.get("name") == name:
                    return child["fileId"]
        r.raise_for_status()
        return r.json()["fileId"]

    def upload_file(
        self,
        filename: str,
        content: bytes | str,
        parent_id: Optional[str] = None,
        specific_space_id: Optional[str] = None,
    ) -> str:
        """
        Create a new file and upload its content.

        Parameters
        ----------
        filename : str
            Name of the file to create.
        content : bytes | str
            File content.  Strings are encoded as UTF-8.
        parent_id : str, optional
            GUID of the parent directory.  Defaults to ``self.space_id``
            (space root).
        specific_space_id : str, optional
            GUID of the specific space where the file should be uploaded.
            If not provided, the default space ID is used.

        Returns
        -------
        str  — the new file's GUID.

        Raises
        ------
        requests.HTTPError  on any non-2xx response.
        """
        if isinstance(content, str):
            content = content.encode("utf-8")

        pid = parent_id or specific_space_id or self.space_id
        base = self._base_url
        # Step 1: create the file node
        create_url = (
            f"{base}{self.PROVIDER_BASE}/data/{pid}/children?name={filename}&type=REG"
        )

        r = self._session.post(create_url, timeout=(10, 30))
        r.raise_for_status()
        file_id = r.json().get("fileId", "")
        if not file_id:
            raise RuntimeError(
                f"File created but no fileId returned.  Response: {r.text!r}"
            )
        # print(f"Created file {filename!r} with ID: {file_id}")

        # Step 2: upload the content
        upload_url = f"{base}{self.PROVIDER_BASE}/data/{file_id}/content"
        r2 = self._session.put(
            upload_url,
            data=content,
            headers={"Content-Type": "application/octet-stream"},
            timeout=(10, 60),
        )
        r2.raise_for_status()

        # logger.info("Uploaded %r (%d bytes)  fileId=%s", filename, len(content), file_id)
        return file_id

    def delete_file(self, file_id: str) -> None:
        """
        Permanently delete a file by its GUID.

        Parameters
        ----------
        file_id : str
            GUID of the file to delete.

        Raises
        ------
        requests.HTTPError  on any non-2xx response.
        """
        guid = self.extract_guid(file_id)
        url = f"{self._base_url}{self.PROVIDER_BASE}/data/{guid}"
        r = self._session.delete(url, timeout=(10, 30))
        r.raise_for_status()
        # logger.info("Deleted file %s", guid)

    def read_file(self, file_id: str) -> bytes:
        """
        Download the content of a file through this provider.

        Blocks the provider does not store are fetched on the fly from a
        provider that does, as for any other read.

        Parameters
        ----------
        file_id : str
            GUID of the file to read.

        Raises
        ------
        requests.HTTPError  on any non-2xx response.
        """
        guid = self.extract_guid(file_id)
        url = f"{self._base_url}{self.PROVIDER_BASE}/data/{guid}/content"
        r = self._session.get(url, timeout=(10, 60))
        r.raise_for_status()
        return r.content

    def has_file(self, file_id: str) -> bool:
        """
        Return True when a file exists, False when it does not.

        Parameters
        ----------
        file_id : str
            GUID of the file to check. Values with ``guid#`` / ``ch#``
            prefixes are accepted.

        Raises
        ------
        requests.HTTPError
            For any non-404 HTTP error.
        """
        guid = self.extract_guid(file_id)
        url = f"{self._base_url}{self.PROVIDER_BASE}/data/{guid}"
        r = self._session.get(url, timeout=(10, 30))
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return True

    def get_file_size(self, file_id: str) -> int:
        """
        Return the file size in bytes.

        Uses GET /data/{file_id}?attributes=size and extracts the
        ``size`` attribute from the response.
        """
        guid = self.extract_guid(file_id)
        url = f"{self._base_url}{self.PROVIDER_BASE}/data/{guid}"
        data = self._raw_get(url, params={"attributes": "size"})

        size = data.get("size")
        if size is None and isinstance(data.get("attributes"), dict):
            size = data["attributes"].get("size")
        if size is None:
            raise RuntimeError(
                f"No 'size' attribute returned for file {guid!r}. Response: {data}"
            )

        try:
            return int(size)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid 'size' value for file {guid!r}: {size!r}"
            ) from exc

    def get_file_path(self, file_id: str) -> str | None:
        """Return the file path (not full path) for *file_id*, or None on any error.
    
        Uses ``GET /data/{id}?attributes=path``.  Failures (e.g. the file has been
        deleted) are caught and logged so callers don't need to guard against them.
        """
        try:
            guid = self.extract_guid(file_id)
            url = f"{self._base_url}{self.PROVIDER_BASE}/data/{guid}"
            data = self._raw_get(url, params={"attributes": "path"})
            return data.get("path")
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Could not fetch path for file %s: %s", file_id, exc)
            return None

    def get_file_mtime(self, file_id: str) -> float | None:
        """Return the last modification time (unix seconds) of *file_id*, or None on any error.

        Uses ``GET /data/{id}?attributes=mtime``.
        """
        try:
            guid = self.extract_guid(file_id)
            url = f"{self._base_url}{self.PROVIDER_BASE}/data/{guid}"
            data = self._raw_get(url, params={"attributes": "mtime"})
            mtime = data.get("mtime")
            return float(mtime) if mtime is not None else None
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Could not fetch mtime for file %s: %s", file_id, exc)
            return None

    def get_file_name(self, file_id: str) -> str | None:
        """Return the file name (not full path) for *file_id*, or None on any error.

        Uses ``GET /data/{id}?attributes=name``.  Failures (e.g. the file has been
        deleted) are caught and logged so callers don't need to guard against them.
        """
        try:
            guid = self.extract_guid(file_id)
            url = f"{self._base_url}{self.PROVIDER_BASE}/data/{guid}"
            data = self._raw_get(url, params={"attributes": "name"})
            return data.get("name")
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Could not fetch name for file %s: %s", file_id, exc)
            return None

    def get_file_id_from_path(self, path: str) -> str | None:
        """Return the file ID for a given file path, or None on any error.

        Uses ``POST /lookup-file-id/{path}``.  Failures are caught and logged so callers don't need to guard against them.
        """
        try:
            url = f"{self._base_url}{self.PROVIDER_BASE}/lookup-file-id{path}"
            logger.debug("Fetching file ID for path %s via %s", path, url)
            data = self._raw_post(url)
            return data.get("fileId")
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Could not fetch file ID for path %s: %s", path, exc)
            return None

    def get_all_file_ids(self) -> list[str]:
        """Return the IDs of all regular files currently in the configured space.

        The space ID is the root directory.  Children are traversed recursively,
        following continuation tokens when a directory listing is paginated.
        Directory IDs are used only as traversal roots and are not returned.
        """
        if not self.space_id:
            raise RuntimeError("space_id is required to list files")

        file_ids: list[str] = []
        directories = [self.space_id]
        visited_directories: set[str] = set()

        while directories:
            directory_id = directories.pop()
            if directory_id in visited_directories:
                continue
            visited_directories.add(directory_id)

            page_token = None
            while True:
                params = {
                    "attribute": ["name", "type", "file_id"],
                    "limit": 1000,
                }
                if page_token:
                    params["token"] = page_token

                url = (
                    f"{self._provider_api_base}{self.PROVIDER_BASE}"
                    f"/data/{directory_id}/children"
                )
                payload = self._raw_get(url, params=params)
                children = payload.get("children", [])

                for child in children:
                    child_id = child.get("file_id") or child.get("fileId")
                    if not child_id:
                        continue
                    child_type = str(child.get("type", "")).upper()
                    if child_type in {"DIR", "DIRECTORY"}:
                        directories.append(child_id)
                    else:
                        file_ids.append(child_id)

                page_token = payload.get("nextPageToken")
                if payload.get("isLast", True) or not page_token:
                    break

        return list(dict.fromkeys(file_ids))

    def get_file_popularity(self, zone_manager: StorageInterface) -> dict:
        """
        Query the file-popularity view for the configured space on each provider.

        Endpoint format per provider:
            https://<provider-hostname>:443/api/v3/oneprovider/spaces/<space_id>/views/file-popularity/query

        Returns
        -------
        dict
            Mapping of provider_id -> list of `value` entries sorted by
            descending popularity score (`key`, higher is more popular).
        """
        if not self.space_id:
            raise RuntimeError("space_id is required to query file popularity")

        if zone_manager is None:
            raise RuntimeError("zone_manager is required to query file popularity")

        providers = zone_manager.list_providers()

        popularity_by_provider = {}
        for provider in providers:
            hostname = provider.domain.strip()
            if not hostname:
                logger.debug(
                    "Skipping provider %s (%s): empty domain",
                    provider.provider_id,
                    provider.name,
                )
                continue

            url = (
                f"https://{hostname}:443"
                f"{self.PROVIDER_BASE}/spaces/{self.space_id}/views/file-popularity/query"
            )
            response = self._session.get(url, timeout=(10, 30))
            response.raise_for_status()

            payload = response.json()
            rows = payload if isinstance(payload, list) else payload.get("rows", [])

            sorted_values = [
                entry.get("value")
                for entry in sorted(
                    (item for item in rows if isinstance(item, dict)),
                    key=lambda item: item.get("key", float("-inf")),
                    reverse=True,
                )
                if entry.get("value") is not None
            ]
            logger.info(
                "Provider %s (%s): retrieved %d popularity entries",
                provider.provider_id,
                provider.name,
                len(sorted_values),
            )
            popularity_by_provider[provider.provider_id] = sorted_values

        return popularity_by_provider

    def get_file_popularity_with_values(self, zone_manager: StorageInterface) -> dict:
        """
        Query the file-popularity view for the configured space on each provider.

        Endpoint format per provider:
            https://<provider-hostname>:443/api/v3/oneprovider/spaces/<space_id>/views/file-popularity/query

        Returns
        -------
        dict
            Mapping of provider_id -> list of `value` entries sorted by
            descending popularity score (`key`, higher is more popular).
        """
        if not self.space_id:
            raise RuntimeError("space_id is required to query file popularity")

        if zone_manager is None:
            raise RuntimeError("zone_manager is required to query file popularity")

        providers = zone_manager.list_providers()

        popularity_by_provider = {}
        for provider in providers:
            hostname = provider.domain.strip()
            if not hostname:
                logger.debug(
                    "Skipping provider %s (%s): empty domain",
                    provider.provider_id,
                    provider.name,
                )
                continue

            url = (
                f"https://{hostname}:443"
                f"{self.PROVIDER_BASE}/spaces/{self.space_id}/views/file-popularity/query"
            )
            response = self._session.get(url, timeout=(10, 30))
            response.raise_for_status()

            payload = response.json()
            rows = payload if isinstance(payload, list) else payload.get("rows", [])

            popularity_values = [
                {
                    "key": entry.get("key"),
                    "value": entry.get("value"),
                }
                for entry in sorted(
                    (item for item in rows if isinstance(item, dict)),
                    key=lambda item: item.get("key", float("-inf")),
                    reverse=True,
                )
                if entry.get("value") is not None and entry.get("key") is not None
            ]

            logger.info(
                "Provider %s (%s): retrieved %d popularity entries",
                provider.provider_id,
                provider.name,
                len(popularity_values),
            )
            popularity_by_provider[provider.provider_id] = popularity_values

        return popularity_by_provider

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _op_url(self, file_ref: str) -> str:
        base = self._provider_api_base
        if self.use_guid:
            guid = self.extract_guid(file_ref)
            return f"{base}{self.PROVIDER_BASE}/data/{guid}/distribution"
        full = f"{self.space_name}/{file_ref}" if self.space_name else file_ref
        return f"{base}{self.PROVIDER_BASE}/replicas/{full}"

    def _raw_get(self, url: str, params: Optional[dict] = None) -> dict:
        r = self._session.get(url, params=params, timeout=(10, 30))
        r.raise_for_status()
        return r.json()

    def _raw_post(
        self,
        url: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
    ) -> dict:
        r = self._session.post(url, params=params, json=body or {}, timeout=(10, 30))
        r.raise_for_status()
        return r.json()

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update(self.config.auth_headers)
        s.verify = self.config.verify_ssl
        return s
