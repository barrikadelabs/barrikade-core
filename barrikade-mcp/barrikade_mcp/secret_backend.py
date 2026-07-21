import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx


@dataclass
class ScopedToken:
    credential: str      # the actual token (was your f-string)
    lease_id: str        # the revocation handle (OpenBao's "accessor")
    expires_at: float    # unix time it dies


class SecretBackend(Protocol):
    capability_grade: str                      # "hard-lease" | "expiry-only" | "static"
    def mint(self, scope: str, ttl_seconds: int) -> ScopedToken: ...
    def revoke(self, lease_id: str) -> None: ...


class OpenBaoBackend:
    capability_grade = "hard-lease"            # OpenBao enforces TTL + real revocation

    def __init__(self):
        self.addr = os.environ.get("OPENBAO_ADDR", "http://127.0.0.1:8200")  # address default: harmless
        self.token = os.environ.get("OPENBAO_TOKEN")                          # credential default: never
        if not self.token:
            raise RuntimeError("OPENBAO_TOKEN is not set (the backend needs a store token to mint)")

    def _headers(self):
        return {"X-Vault-Token": self.token}

    def mint(self, scope: str, ttl_seconds: int) -> ScopedToken:
        # POST {addr}/v1/auth/token/create  body: ttl="<n>s", meta={"scope": scope}, no_parent=True
        # from the response's "auth" object, read client_token, accessor, lease_duration
        resp = httpx.post(
        f"{self.addr}/v1/auth/token/create",
        headers=self._headers(),
        json={"ttl": f"{ttl_seconds}s", "meta": {"scope": scope}, "no_parent": True},
        )                
        resp.raise_for_status()
        auth = resp.json()["auth"]
        return ScopedToken(
            credential=auth["client_token"],                  
            lease_id=auth["accessor"],                     
            expires_at=time.time() + auth["lease_duration"],
        )

    def revoke(self, lease_id: str) -> None:
        # POST {addr}/v1/auth/token/revoke-accessor  body: accessor=lease_id
        resp = httpx.post(
        f"{self.addr}/v1/auth/token/revoke-accessor",
        headers=self._headers(),
        json={"accessor": lease_id},
        )
        
        resp.raise_for_status()                      
