"""
Input Validation Module.

Comprehensive input validation and sanitization:
- SQL injection prevention
- XSS prevention
- Command injection prevention
- Path traversal prevention
- Format validation

Defense in depth - validate at multiple layers.
"""

import re
import html
from typing import Optional, Any
from dataclasses import dataclass

from src.common.logging import get_logger

logger = get_logger(__name__)


# Dangerous patterns
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|UNION|FROM|WHERE)\b)",
    r"(--|;|/\*|\*/|@@|@)",
    r"(\bOR\b.*=.*\bOR\b)",
    r"(\bAND\b.*=.*\bAND\b)",
    r"(\'.*\bOR\b.*\')",
]

XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe[^>]*>",
    r"<object[^>]*>",
    r"<embed[^>]*>",
]

COMMAND_INJECTION_PATTERNS = [
    r"[;&|`$]",
    r"\$\(",
    r"`.*`",
    r"\|\|",
    r"&&",
]

PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e%2f",
    r"%2e%2e/",
    r"\.%2e/",
]


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized: Any
    errors: list[str]
    warnings: list[str]


class InputValidator:
    """
    Comprehensive input validation.

    Validates and sanitizes various input types.
    """

    # Email regex (RFC 5322 simplified)
    EMAIL_PATTERN = re.compile(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    )

    # Phone regex (international format)
    PHONE_PATTERN = re.compile(
        r"^\+?[1-9]\d{1,14}$"
    )

    # UUID regex
    UUID_PATTERN = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )

    # H3 index regex (hexadecimal, 15-16 chars)
    H3_PATTERN = re.compile(r"^[0-9a-f]{15}$", re.IGNORECASE)

    # Latitude/Longitude ranges
    LAT_MIN, LAT_MAX = -90.0, 90.0
    LNG_MIN, LNG_MAX = -180.0, 180.0

    @classmethod
    def validate_string(
        cls,
        value: str,
        min_length: int = 0,
        max_length: int = 10000,
        allow_html: bool = False,
        allow_sql_keywords: bool = False,
    ) -> ValidationResult:
        """
        Validate and sanitize a string input.
        """
        errors = []
        warnings = []

        if not isinstance(value, str):
            return ValidationResult(False, "", ["Value must be a string"], [])

        # Length checks
        if len(value) < min_length:
            errors.append(f"Value must be at least {min_length} characters")

        if len(value) > max_length:
            errors.append(f"Value must be at most {max_length} characters")
            value = value[:max_length]  # Truncate

        # Check for SQL injection
        if not allow_sql_keywords:
            for pattern in SQL_INJECTION_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    warnings.append("Potentially dangerous SQL pattern detected")
                    # Escape dangerous characters
                    value = re.sub(pattern, "", value, flags=re.IGNORECASE)

        # Check for XSS
        if not allow_html:
            for pattern in XSS_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    warnings.append("Potentially dangerous HTML/script pattern detected")
            # HTML escape
            value = html.escape(value)

        # Check for command injection
        for pattern in COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value):
                warnings.append("Potentially dangerous command pattern detected")
                value = re.sub(pattern, "", value)

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized=value,
            errors=errors,
            warnings=warnings,
        )

    @classmethod
    def validate_email(cls, email: str) -> ValidationResult:
        """Validate email address."""
        errors = []

        if not email:
            errors.append("Email is required")
            return ValidationResult(False, "", errors, [])

        email = email.strip().lower()

        if len(email) > 254:
            errors.append("Email is too long")

        if not cls.EMAIL_PATTERN.match(email):
            errors.append("Invalid email format")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized=email,
            errors=errors,
            warnings=[],
        )

    @classmethod
    def validate_phone(cls, phone: str) -> ValidationResult:
        """Validate phone number."""
        errors = []

        if not phone:
            errors.append("Phone number is required")
            return ValidationResult(False, "", errors, [])

        # Remove common formatting
        phone = re.sub(r"[\s\-\(\)\.]", "", phone)

        if not cls.PHONE_PATTERN.match(phone):
            errors.append("Invalid phone number format")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized=phone,
            errors=errors,
            warnings=[],
        )

    @classmethod
    def validate_uuid(cls, value: str) -> ValidationResult:
        """Validate UUID format."""
        errors = []

        if not value:
            errors.append("UUID is required")
            return ValidationResult(False, "", errors, [])

        value = value.strip().lower()

        if not cls.UUID_PATTERN.match(value):
            errors.append("Invalid UUID format")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized=value,
            errors=errors,
            warnings=[],
        )

    @classmethod
    def validate_h3_index(cls, value: str) -> ValidationResult:
        """Validate H3 hexagon index."""
        errors = []

        if not value:
            errors.append("H3 index is required")
            return ValidationResult(False, "", errors, [])

        value = value.strip().lower()

        if not cls.H3_PATTERN.match(value):
            errors.append("Invalid H3 index format")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized=value,
            errors=errors,
            warnings=[],
        )

    @classmethod
    def validate_coordinates(
        cls,
        latitude: float,
        longitude: float,
    ) -> ValidationResult:
        """Validate geographic coordinates."""
        errors = []

        try:
            lat = float(latitude)
            lng = float(longitude)
        except (TypeError, ValueError):
            errors.append("Coordinates must be numeric")
            return ValidationResult(False, (0.0, 0.0), errors, [])

        if not (cls.LAT_MIN <= lat <= cls.LAT_MAX):
            errors.append(f"Latitude must be between {cls.LAT_MIN} and {cls.LAT_MAX}")

        if not (cls.LNG_MIN <= lng <= cls.LNG_MAX):
            errors.append(f"Longitude must be between {cls.LNG_MIN} and {cls.LNG_MAX}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized=(lat, lng),
            errors=errors,
            warnings=[],
        )

    @classmethod
    def validate_positive_number(
        cls,
        value: Any,
        min_value: float = 0,
        max_value: float = float('inf'),
    ) -> ValidationResult:
        """Validate positive number."""
        errors = []

        try:
            num = float(value)
        except (TypeError, ValueError):
            errors.append("Value must be a number")
            return ValidationResult(False, 0, errors, [])

        if num < min_value:
            errors.append(f"Value must be at least {min_value}")

        if num > max_value:
            errors.append(f"Value must be at most {max_value}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized=num,
            errors=errors,
            warnings=[],
        )

    @classmethod
    def validate_path(cls, path: str) -> ValidationResult:
        """Validate file path (prevent traversal attacks)."""
        errors = []
        warnings = []

        if not path:
            errors.append("Path is required")
            return ValidationResult(False, "", errors, [])

        # Check for path traversal
        for pattern in PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                errors.append("Path traversal detected")
                return ValidationResult(False, "", errors, [])

        # Normalize path
        import os
        normalized = os.path.normpath(path)

        if normalized != path:
            warnings.append("Path was normalized")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized=normalized,
            errors=errors,
            warnings=warnings,
        )


# Convenience functions

def sanitize_input(value: str, max_length: int = 1000) -> str:
    """Sanitize string input (convenience function)."""
    result = InputValidator.validate_string(value, max_length=max_length)
    return result.sanitized


def validate_email(email: str) -> tuple[bool, str]:
    """Validate email (convenience function)."""
    result = InputValidator.validate_email(email)
    return result.is_valid, result.sanitized


def validate_phone(phone: str) -> tuple[bool, str]:
    """Validate phone (convenience function)."""
    result = InputValidator.validate_phone(phone)
    return result.is_valid, result.sanitized


def validate_coordinates(lat: float, lng: float) -> tuple[bool, tuple[float, float]]:
    """Validate coordinates (convenience function)."""
    result = InputValidator.validate_coordinates(lat, lng)
    return result.is_valid, result.sanitized


class SQLParameterizer:
    """
    Helper for safe SQL parameter handling.

    Always use parameterized queries!
    """

    @staticmethod
    def escape_like(value: str) -> str:
        """Escape LIKE pattern special characters."""
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @staticmethod
    def quote_identifier(identifier: str) -> str:
        """Quote a SQL identifier (table/column name)."""
        # Validate identifier (alphanumeric and underscore only)
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier}")
        return f'"{identifier}"'

