"""
Tests for Security Module.

Comprehensive test suite covering:
- Authentication (JWT, password hashing)
- Authorization (RBAC, permissions)
- Rate limiting
- Input validation
- Audit logging
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.security.auth import (
    PasswordHandler,
    JWTHandler,
    AuthenticationService,
    TokenPayload,
    TokenType,
    create_access_token,
    verify_access_token,
)
from src.common.security.authorization import (
    AuthorizationService,
    Permission,
    Role,
    ROLE_PERMISSIONS,
)
from src.common.security.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    TokenBucketLimiter,
    SlidingWindowLimiter,
    RateLimitExceeded,
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
    SecurityEventType,
    SeverityLevel,
)


# ============== Password Handler Tests ==============

class TestPasswordHandler:
    """Tests for password hashing and validation."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "SecureP@ssw0rd!"
        hashed = PasswordHandler.hash(password)

        assert hashed != password
        assert len(hashed) > 20

    def test_verify_password_correct(self):
        """Test correct password verification."""
        password = "SecureP@ssw0rd!"
        hashed = PasswordHandler.hash(password)

        assert PasswordHandler.verify(password, hashed) == True

    def test_verify_password_incorrect(self):
        """Test incorrect password verification."""
        password = "SecureP@ssw0rd!"
        hashed = PasswordHandler.hash(password)

        assert PasswordHandler.verify("WrongPassword!", hashed) == False

    def test_password_strength_valid(self):
        """Test valid password strength."""
        password = "SecureP@ssw0rd!"
        is_valid, errors = PasswordHandler.validate_strength(password)

        assert is_valid == True
        assert len(errors) == 0

    def test_password_strength_too_short(self):
        """Test password too short."""
        password = "Sh0rt!"
        is_valid, errors = PasswordHandler.validate_strength(password)

        assert is_valid == False
        assert any("at least 8 characters" in e for e in errors)

    def test_password_strength_missing_uppercase(self):
        """Test password missing uppercase."""
        password = "nouppercase123!"
        is_valid, errors = PasswordHandler.validate_strength(password)

        assert is_valid == False
        assert any("uppercase" in e for e in errors)

    def test_password_strength_missing_special(self):
        """Test password missing special character."""
        password = "NoSpecialChar123"
        is_valid, errors = PasswordHandler.validate_strength(password)

        assert is_valid == False
        assert any("special" in e for e in errors)


# ============== JWT Handler Tests ==============

class TestJWTHandler:
    """Tests for JWT token handling."""

    @pytest.fixture
    def jwt_handler(self):
        return JWTHandler(
            secret_key="test-secret-key-for-testing-only",
            access_token_expire_minutes=15,
        )

    def test_create_access_token(self, jwt_handler):
        """Test access token creation."""
        token = jwt_handler.create_access_token(
            subject="user123",
            roles=["customer"],
        )

        assert isinstance(token, str)
        assert len(token) > 50

    def test_verify_access_token(self, jwt_handler):
        """Test access token verification."""
        token = jwt_handler.create_access_token(
            subject="user123",
            roles=["customer"],
            permissions=["order:create"],
        )

        payload = jwt_handler.verify_token(token, TokenType.ACCESS)

        assert payload.sub == "user123"
        assert "customer" in payload.roles
        assert "order:create" in payload.permissions

    def test_create_refresh_token(self, jwt_handler):
        """Test refresh token creation."""
        token = jwt_handler.create_refresh_token(subject="user123")

        payload = jwt_handler.verify_token(token, TokenType.REFRESH)

        assert payload.sub == "user123"
        assert payload.type == TokenType.REFRESH

    def test_token_type_mismatch(self, jwt_handler):
        """Test that wrong token type raises error."""
        access_token = jwt_handler.create_access_token(subject="user123")

        with pytest.raises(Exception):  # HTTPException
            jwt_handler.verify_token(access_token, TokenType.REFRESH)

    def test_revoke_token(self, jwt_handler):
        """Test token revocation."""
        token = jwt_handler.create_access_token(subject="user123")
        payload = jwt_handler.verify_token(token)

        jwt_handler.revoke_token(payload.jti)

        with pytest.raises(Exception):  # HTTPException
            jwt_handler.verify_token(token)


# ============== Authorization Tests ==============

class TestAuthorizationService:
    """Tests for authorization service."""

    @pytest.fixture
    def auth_service(self):
        return AuthorizationService()

    @pytest.fixture
    def admin_user(self):
        return TokenPayload(
            sub="admin1",
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="jti1",
            roles=["admin"],
            permissions=[],
        )

    @pytest.fixture
    def customer_user(self):
        return TokenPayload(
            sub="customer1",
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="jti2",
            roles=["customer"],
            permissions=[],
        )

    def test_admin_has_all_permissions(self, auth_service, admin_user):
        """Test that admin has all permissions."""
        for permission in Permission:
            assert auth_service.has_permission(admin_user, permission) == True

    def test_customer_permissions(self, auth_service, customer_user):
        """Test customer permissions."""
        # Customer should have these
        assert auth_service.has_permission(customer_user, Permission.ORDER_CREATE) == True
        assert auth_service.has_permission(customer_user, Permission.ORDER_READ) == True

        # Customer should NOT have these
        assert auth_service.has_permission(customer_user, Permission.ADMIN_SYSTEM) == False
        assert auth_service.has_permission(customer_user, Permission.DISPATCH_RUN) == False

    def test_role_check(self, auth_service, admin_user, customer_user):
        """Test role checking."""
        assert auth_service.has_role(admin_user, Role.ADMIN) == True
        assert auth_service.has_role(admin_user, Role.CUSTOMER) == False

        assert auth_service.has_role(customer_user, Role.CUSTOMER) == True
        assert auth_service.has_role(customer_user, Role.ADMIN) == False

    def test_explicit_permissions(self, auth_service):
        """Test explicit permissions override roles."""
        user = TokenPayload(
            sub="user1",
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="jti3",
            roles=["customer"],
            permissions=["admin:system"],  # Explicit permission
        )

        assert auth_service.has_permission(user, Permission.ADMIN_SYSTEM) == True


# ============== Rate Limiter Tests ==============

class TestTokenBucketLimiter:
    """Tests for token bucket rate limiting."""

    def test_allows_under_limit(self):
        """Test requests allowed under limit."""
        config = RateLimitConfig(requests=10, window_seconds=60)
        limiter = TokenBucketLimiter(config)

        for i in range(10):
            allowed, _ = limiter.is_allowed("client1")
            assert allowed == True

    def test_blocks_over_limit(self):
        """Test requests blocked over limit."""
        config = RateLimitConfig(requests=5, window_seconds=60, burst=5)
        limiter = TokenBucketLimiter(config)

        # Exhaust tokens
        for i in range(5):
            limiter.is_allowed("client1")

        # Next should be blocked
        allowed, metadata = limiter.is_allowed("client1")
        assert allowed == False
        assert "retry_after" in metadata

    def test_different_clients_independent(self):
        """Test that different clients have independent limits."""
        config = RateLimitConfig(requests=2, window_seconds=60, burst=2)
        limiter = TokenBucketLimiter(config)

        # Exhaust client1
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        allowed1, _ = limiter.is_allowed("client1")

        # Client2 should still be allowed
        allowed2, _ = limiter.is_allowed("client2")

        assert allowed1 == False
        assert allowed2 == True


class TestSlidingWindowLimiter:
    """Tests for sliding window rate limiting."""

    def test_allows_under_limit(self):
        """Test requests allowed under limit."""
        config = RateLimitConfig(requests=5, window_seconds=60)
        limiter = SlidingWindowLimiter(config)

        for i in range(5):
            allowed, _ = limiter.is_allowed("client1")
            assert allowed == True

    def test_blocks_over_limit(self):
        """Test requests blocked over limit."""
        config = RateLimitConfig(requests=3, window_seconds=60)
        limiter = SlidingWindowLimiter(config)

        for i in range(3):
            limiter.is_allowed("client1")

        allowed, _ = limiter.is_allowed("client1")
        assert allowed == False


class TestRateLimiter:
    """Tests for main rate limiter."""

    def test_default_limits_exist(self):
        """Test default limit configurations exist."""
        limiter = RateLimiter()

        assert "global" in limiter.DEFAULT_LIMITS
        assert "auth" in limiter.DEFAULT_LIMITS
        assert "api" in limiter.DEFAULT_LIMITS

    def test_check_and_raise(self):
        """Test check_and_raise raises exception."""
        limiter = RateLimiter()

        # Create strict limit
        limiter._limiters["test"] = TokenBucketLimiter(
            RateLimitConfig(requests=1, window_seconds=60, burst=1)
        )

        # First should pass
        limiter.check_and_raise("client", "test")

        # Second should raise
        with pytest.raises(RateLimitExceeded):
            limiter.check_and_raise("client", "test")


# ============== Input Validator Tests ==============

class TestInputValidator:
    """Tests for input validation."""

    def test_validate_string_basic(self):
        """Test basic string validation."""
        result = InputValidator.validate_string("Hello World")

        assert result.is_valid == True
        assert result.sanitized == "Hello World"

    def test_validate_string_too_long(self):
        """Test string too long."""
        result = InputValidator.validate_string("a" * 200, max_length=100)

        assert result.is_valid == False
        assert len(result.sanitized) == 100

    def test_validate_string_xss_prevention(self):
        """Test XSS prevention."""
        result = InputValidator.validate_string("<script>alert('xss')</script>")

        # Should not contain raw script tags (either escaped or removed)
        assert "<script>" not in result.sanitized
        # Result should be sanitized in some form
        assert result.sanitized != "<script>alert('xss')</script>"

    def test_validate_email_valid(self):
        """Test valid email."""
        result = InputValidator.validate_email("user@example.com")

        assert result.is_valid == True
        assert result.sanitized == "user@example.com"

    def test_validate_email_invalid(self):
        """Test invalid email."""
        result = InputValidator.validate_email("not-an-email")

        assert result.is_valid == False

    def test_validate_phone_valid(self):
        """Test valid phone number."""
        result = InputValidator.validate_phone("+79001234567")

        assert result.is_valid == True

    def test_validate_phone_invalid(self):
        """Test invalid phone number."""
        result = InputValidator.validate_phone("abc123")

        assert result.is_valid == False

    def test_validate_coordinates_valid(self):
        """Test valid coordinates."""
        result = InputValidator.validate_coordinates(55.7558, 37.6173)

        assert result.is_valid == True
        assert result.sanitized == (55.7558, 37.6173)

    def test_validate_coordinates_out_of_range(self):
        """Test coordinates out of range."""
        result = InputValidator.validate_coordinates(100.0, 37.6173)

        assert result.is_valid == False

    def test_validate_uuid_valid(self):
        """Test valid UUID."""
        result = InputValidator.validate_uuid("550e8400-e29b-41d4-a716-446655440000")

        assert result.is_valid == True

    def test_validate_uuid_invalid(self):
        """Test invalid UUID."""
        result = InputValidator.validate_uuid("not-a-uuid")

        assert result.is_valid == False

    def test_validate_path_traversal_blocked(self):
        """Test path traversal is blocked."""
        result = InputValidator.validate_path("../../../etc/passwd")

        assert result.is_valid == False

    def test_sql_injection_detection(self):
        """Test SQL injection detection."""
        result = InputValidator.validate_string(
            "'; DROP TABLE users; --",
            allow_sql_keywords=False,
        )

        assert len(result.warnings) > 0


# ============== Audit Logger Tests ==============

class TestAuditLogger:
    """Tests for security audit logging."""

    @pytest.fixture
    def audit_logger(self):
        return AuditLogger()

    def test_log_event(self, audit_logger):
        """Test logging a security event."""
        event = SecurityEvent(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            severity=SeverityLevel.INFO,
            user_id="user123",
            ip_address="192.168.1.1",
            message="User logged in",
        )

        audit_logger.log(event)

        counts = audit_logger.get_event_counts()
        assert counts[SecurityEventType.LOGIN_SUCCESS.value] == 1

    def test_log_login_success(self, audit_logger):
        """Test convenience method for login success."""
        audit_logger.log_login_success(
            user_id="user123",
            ip_address="192.168.1.1",
        )

        counts = audit_logger.get_event_counts()
        assert counts[SecurityEventType.LOGIN_SUCCESS.value] == 1

    def test_log_login_failure(self, audit_logger):
        """Test convenience method for login failure."""
        audit_logger.log_login_failure(
            username="baduser",
            ip_address="192.168.1.1",
            reason="Invalid password",
        )

        counts = audit_logger.get_event_counts()
        assert counts[SecurityEventType.LOGIN_FAILURE.value] == 1

    def test_event_to_json(self):
        """Test event serialization to JSON."""
        event = SecurityEvent(
            event_type=SecurityEventType.ACCESS_DENIED,
            severity=SeverityLevel.WARNING,
            user_id="user123",
            resource_type="order",
            resource_id="order456",
        )

        json_str = event.to_json()

        # Check that it contains expected data
        assert "authz.access.denied" in json_str  # Full event type value
        assert "user123" in json_str
        assert "order456" in json_str

    def test_event_fingerprint(self):
        """Test event fingerprint for deduplication."""
        event1 = SecurityEvent(
            event_type=SecurityEventType.LOGIN_FAILURE,
            severity=SeverityLevel.WARNING,
            user_id="user123",
            ip_address="192.168.1.1",
        )

        event2 = SecurityEvent(
            event_type=SecurityEventType.LOGIN_FAILURE,
            severity=SeverityLevel.WARNING,
            user_id="user123",
            ip_address="192.168.1.1",
        )

        # Same fingerprint for same key attributes
        assert event1.fingerprint == event2.fingerprint

    def test_custom_handler(self, audit_logger):
        """Test custom event handler."""
        events_received = []

        def custom_handler(event):
            events_received.append(event)

        audit_logger.add_handler(custom_handler)

        audit_logger.log_login_success("user123", "192.168.1.1")

        assert len(events_received) == 1
        assert events_received[0].user_id == "user123"


# ============== Convenience Function Tests ==============

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_sanitize_input(self):
        """Test sanitize_input function."""
        result = sanitize_input("<script>alert(1)</script>")
        assert "<script>" not in result

    def test_validate_email_function(self):
        """Test validate_email function."""
        is_valid, sanitized = validate_email("USER@Example.COM")

        assert is_valid == True
        assert sanitized == "user@example.com"

    def test_validate_phone_function(self):
        """Test validate_phone function."""
        is_valid, sanitized = validate_phone("+79001234567")

        assert is_valid == True
        assert sanitized == "+79001234567"

    def test_validate_coordinates_function(self):
        """Test validate_coordinates function."""
        is_valid, coords = validate_coordinates(55.7558, 37.6173)

        assert is_valid == True
        assert coords == (55.7558, 37.6173)

