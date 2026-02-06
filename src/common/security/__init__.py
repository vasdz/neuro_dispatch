"""
Security Module.

Production-grade security infrastructure for the NeuroDispatch platform.

Components:
- Authentication: JWT-based token authentication
- Authorization: Role-based access control (RBAC)
- Rate Limiting: Protection against abuse
- Input Validation: Defense against injection attacks
- Audit Logging: Security event tracking
- Encryption: Data protection utilities

Senior+ security best practices implemented.
"""

from src.common.security.auth import (
    AuthenticationService,
    JWTHandler,
    PasswordHandler,
    TokenPayload,
    create_access_token,
    verify_access_token,
    get_current_user,
    get_current_active_user,
)
from src.common.security.authorization import (
    AuthorizationService,
    Permission,
    Role,
    require_permission,
    require_role,
)
from src.common.security.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    rate_limit,
)
from src.common.security.validators import (
    InputValidator,
    sanitize_input,
    validate_email,
    validate_phone,
    validate_coordinates,
)
from src.common.security.audit import (
    AuditLogger,
    SecurityEvent,
    log_security_event,
)

__all__ = [
    # Authentication
    "AuthenticationService",
    "JWTHandler",
    "PasswordHandler",
    "TokenPayload",
    "create_access_token",
    "verify_access_token",
    "get_current_user",
    "get_current_active_user",
    # Authorization
    "AuthorizationService",
    "Permission",
    "Role",
    "require_permission",
    "require_role",
    # Rate Limiting
    "RateLimiter",
    "RateLimitExceeded",
    "rate_limit",
    # Validation
    "InputValidator",
    "sanitize_input",
    "validate_email",
    "validate_phone",
    "validate_coordinates",
    # Audit
    "AuditLogger",
    "SecurityEvent",
    "log_security_event",
]

