"""
Security Audit Logging Module.

Comprehensive security event logging for:
- Authentication events (login, logout, failures)
- Authorization events (access granted/denied)
- Data access events (sensitive data views)
- Admin actions (configuration changes)
- Security anomalies (suspicious patterns)

Compliant with:
- GDPR audit requirements
- SOC 2 logging requirements
- PCI-DSS logging requirements
"""

from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import json
import hashlib

from src.common.logging import get_logger

logger = get_logger(__name__)


class SecurityEventType(str, Enum):
    """Types of security events."""
    # Authentication
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token.refresh"
    TOKEN_REVOKE = "auth.token.revoke"
    PASSWORD_CHANGE = "auth.password.change"
    PASSWORD_RESET_REQUEST = "auth.password.reset.request"
    MFA_ENABLED = "auth.mfa.enabled"
    MFA_DISABLED = "auth.mfa.disabled"

    # Authorization
    ACCESS_GRANTED = "authz.access.granted"
    ACCESS_DENIED = "authz.access.denied"
    PERMISSION_CHANGE = "authz.permission.change"
    ROLE_CHANGE = "authz.role.change"

    # Data Access
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    SENSITIVE_DATA_ACCESS = "data.sensitive.access"

    # Admin Actions
    USER_CREATED = "admin.user.created"
    USER_DELETED = "admin.user.deleted"
    USER_MODIFIED = "admin.user.modified"
    CONFIG_CHANGED = "admin.config.changed"
    SYSTEM_SETTING_CHANGED = "admin.system.changed"

    # Security Anomalies
    RATE_LIMIT_EXCEEDED = "security.ratelimit.exceeded"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    BRUTE_FORCE_DETECTED = "security.bruteforce"
    INJECTION_ATTEMPT = "security.injection"
    INVALID_TOKEN = "security.token.invalid"

    # System Events
    SERVICE_START = "system.service.start"
    SERVICE_STOP = "system.service.stop"
    ERROR = "system.error"


class SeverityLevel(str, Enum):
    """Severity levels for security events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """
    A security audit event.

    Contains all relevant information for security auditing.
    """
    event_type: SecurityEventType
    severity: SeverityLevel

    # Who
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_roles: list[str] = field(default_factory=list)

    # What
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None

    # Where
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None

    # When
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Details
    message: str = ""
    details: dict = field(default_factory=dict)

    # Result
    success: bool = True
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Correlation
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @property
    def fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        key = f"{self.event_type.value}:{self.user_id}:{self.ip_address}:{self.resource_id}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


class AuditLogger:
    """
    Security audit logger.

    Logs security events to multiple destinations:
    - Structured logs
    - Database (for querying)
    - External SIEM (optional)
    """

    def __init__(self):
        self._handlers: list = []
        self._event_counts: dict[str, int] = {}

    def log(self, event: SecurityEvent):
        """Log a security event."""
        # Update event counts
        self._event_counts[event.event_type.value] = (
            self._event_counts.get(event.event_type.value, 0) + 1
        )

        # Log to structured logger
        log_func = self._get_log_function(event.severity)
        log_func(
            event.message or event.event_type.value,
            event_type=event.event_type.value,
            user_id=event.user_id,
            resource_type=event.resource_type,
            resource_id=event.resource_id,
            ip_address=event.ip_address,
            success=event.success,
            **event.details,
        )

        # Call custom handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Audit handler error: {e}")

    def _get_log_function(self, severity: SeverityLevel):
        """Get the appropriate log function for severity."""
        return {
            SeverityLevel.DEBUG: logger.debug,
            SeverityLevel.INFO: logger.info,
            SeverityLevel.WARNING: logger.warning,
            SeverityLevel.ERROR: logger.error,
            SeverityLevel.CRITICAL: logger.critical,
        }.get(severity, logger.info)

    def add_handler(self, handler):
        """Add a custom event handler."""
        self._handlers.append(handler)

    def get_event_counts(self) -> dict[str, int]:
        """Get event counts for monitoring."""
        return self._event_counts.copy()

    # Convenience methods for common events

    def log_login_success(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str = None,
    ):
        """Log successful login."""
        self.log(SecurityEvent(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            severity=SeverityLevel.INFO,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            message=f"User {user_id} logged in successfully",
            success=True,
        ))

    def log_login_failure(
        self,
        username: str,
        ip_address: str,
        reason: str = "Invalid credentials",
        user_agent: str = None,
    ):
        """Log failed login attempt."""
        self.log(SecurityEvent(
            event_type=SecurityEventType.LOGIN_FAILURE,
            severity=SeverityLevel.WARNING,
            user_id=username,  # May not be valid user
            ip_address=ip_address,
            user_agent=user_agent,
            message=f"Login failed for {username}: {reason}",
            success=False,
            error_message=reason,
        ))

    def log_access_denied(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        permission: str,
        ip_address: str = None,
    ):
        """Log access denied event."""
        self.log(SecurityEvent(
            event_type=SecurityEventType.ACCESS_DENIED,
            severity=SeverityLevel.WARNING,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            message=f"Access denied for {user_id} to {resource_type}/{resource_id}",
            success=False,
            details={"required_permission": permission},
        ))

    def log_sensitive_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        fields: list[str],
        ip_address: str = None,
    ):
        """Log sensitive data access."""
        self.log(SecurityEvent(
            event_type=SecurityEventType.SENSITIVE_DATA_ACCESS,
            severity=SeverityLevel.INFO,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            message=f"Sensitive data accessed: {resource_type}/{resource_id}",
            success=True,
            details={"fields": fields},
        ))

    def log_rate_limit_exceeded(
        self,
        ip_address: str,
        endpoint: str,
        limit: int,
        user_id: str = None,
    ):
        """Log rate limit exceeded."""
        self.log(SecurityEvent(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            severity=SeverityLevel.WARNING,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            message=f"Rate limit exceeded for {ip_address} on {endpoint}",
            success=False,
            details={"limit": limit},
        ))

    def log_suspicious_activity(
        self,
        description: str,
        ip_address: str,
        user_id: str = None,
        details: dict = None,
    ):
        """Log suspicious activity."""
        self.log(SecurityEvent(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            severity=SeverityLevel.WARNING,
            user_id=user_id,
            ip_address=ip_address,
            message=f"Suspicious activity: {description}",
            success=False,
            details=details or {},
        ))

    def log_config_change(
        self,
        user_id: str,
        config_key: str,
        old_value: Any,
        new_value: Any,
        ip_address: str = None,
    ):
        """Log configuration change."""
        self.log(SecurityEvent(
            event_type=SecurityEventType.CONFIG_CHANGED,
            severity=SeverityLevel.INFO,
            user_id=user_id,
            ip_address=ip_address,
            resource_type="config",
            resource_id=config_key,
            message=f"Configuration changed: {config_key}",
            success=True,
            details={
                "old_value": str(old_value),
                "new_value": str(new_value),
            },
        ))


# Global audit logger instance
_audit_logger = AuditLogger()


def log_security_event(event: SecurityEvent):
    """Log a security event (convenience function)."""
    _audit_logger.log(event)


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger."""
    return _audit_logger

