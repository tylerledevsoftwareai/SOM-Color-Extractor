import os
import logging
from typing import Optional, List
from fastapi import Request, HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger("som_service.auth")

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

class AccessControl:
    """
    Access Control manager that validates Client IP address, Origin/Referer website domains, 
    and API Key headers against environment variable permissions.
    """
    def __init__(self):
        self._reload_permissions()

    def _reload_permissions(self):
        # Parse comma-separated environment variables
        allowed_ips_raw = os.getenv("ALLOWED_IPS", "").strip()
        allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "").strip()
        allowed_keys_raw = os.getenv("ALLOWED_API_KEYS", "").strip()

        self.allowed_ips = set(ip.strip() for ip in allowed_ips_raw.split(",") if ip.strip())
        self.allowed_origins = set(origin.strip().rstrip("/") for origin in allowed_origins_raw.split(",") if origin.strip())
        self.allowed_api_keys = set(key.strip() for key in allowed_keys_raw.split(",") if key.strip())

        # Security mode flags
        self.is_ip_restricted = len(self.allowed_ips) > 0 and "*" not in self.allowed_ips
        self.is_origin_restricted = len(self.allowed_origins) > 0 and "*" not in self.allowed_origins
        self.is_key_restricted = len(self.allowed_api_keys) > 0 and "*" not in self.allowed_api_keys

    def extract_client_ip(self, request: Request) -> str:
        """
        Extract real client IP, considering reverse proxy headers (e.g. Render, Cloudflare).
        """
        x_forwarded_for = request.headers.get("x-forwarded-for")
        if x_forwarded_for:
            # First IP in the comma-separated list is the client IP
            return x_forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
            
        return request.client.host if request.client else "127.0.0.1"

    def extract_origin(self, request: Request) -> Optional[str]:
        """
        Extract client website domain from Origin or Referer headers.
        """
        origin = request.headers.get("origin")
        if origin:
            return origin.strip().rstrip("/")
        
        referer = request.headers.get("referer")
        if referer:
            # Extract scheme + domain from referer URL
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            if parsed.scheme and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}"
        return None

    def verify_request(self, request: Request, api_key: Optional[str] = None) -> bool:
        """
        Verify if an incoming request is authorized based on IP, Origin domain, or API Key.
        If no restrictions are set, all requests are permitted.
        If restrictions are configured, at least one permission check must pass.
        """
        self._reload_permissions()

        # If no security rules are set, allow all traffic
        if not self.is_ip_restricted and not self.is_origin_restricted and not self.is_key_restricted:
            return True

        reasons = []
        
        # 1. API Key check
        if self.is_key_restricted:
            if api_key and api_key in self.allowed_api_keys:
                return True
            reasons.append("Invalid or missing API Key (header 'X-API-Key')")

        # 2. Client IP check
        client_ip = self.extract_client_ip(request)
        if self.is_ip_restricted:
            if client_ip in self.allowed_ips or "127.0.0.1" in self.allowed_ips and client_ip == "localhost":
                return True
            reasons.append(f"Client IP '{client_ip}' is not whitelisted")

        # 3. Website Origin / Referer domain check
        client_origin = self.extract_origin(request)
        if self.is_origin_restricted:
            if client_origin and client_origin in self.allowed_origins:
                return True
            reasons.append(f"Origin website '{client_origin or 'Unknown'}' is not whitelisted")

        # If none of the applicable permission checks passed
        error_msg = "Access Denied: " + "; ".join(reasons)
        logger.warning(f"Forbidden access attempt from IP {client_ip}: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Forbidden",
                "message": "Access denied. Your IP address, website origin, or API key is not authorized.",
                "client_ip": client_ip,
                "client_origin": client_origin,
                "reasons": reasons
            }
        )

# Global singleton access controller
access_control = AccessControl()

async def verify_permissions(request: Request, api_key: Optional[str] = Security(API_KEY_HEADER)):
    """
    FastAPI Security Dependency to enforce authorization on routes.
    """
    access_control.verify_request(request, api_key)
