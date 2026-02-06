"""
Authorization Module.

Implements Role-Based Access Control (RBAC) with:
- Predefined roles (admin, operator, courier, customer)
- Fine-grained permissions
- Resource-level authorization
- Policy-based access control ready

Security best practices:
- Principle of least privilege
- Separation of duties
- Deny by default
"""

from enum import Enum
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional, Callable, Any

from fastapi import Depends, HTTPException, status

from src.common.logging import get_logger
from src.common.security.auth import TokenPayload, get_current_user

logger = get_logger(__name__)


class Permission(str, Enum):
    """
    Granular permissions for the system.

    Format: resource:action
    """
    # Orders
    ORDER_CREATE = "order:create"
    ORDER_READ = "order:read"
    ORDER_UPDATE = "order:update"
    ORDER_DELETE = "order:delete"
    ORDER_ASSIGN = "order:assign"
    ORDER_CANCEL = "order:cancel"

    # Couriers
    COURIER_READ = "courier:read"
    COURIER_UPDATE = "courier:update"
    COURIER_MANAGE = "courier:manage"
    COURIER_LOCATION_READ = "courier:location:read"

    # Restaurants
    RESTAURANT_READ = "restaurant:read"
    RESTAURANT_UPDATE = "restaurant:update"
    RESTAURANT_MANAGE = "restaurant:manage"

    # Dispatch
    DISPATCH_RUN = "dispatch:run"
    DISPATCH_CONFIGURE = "dispatch:configure"

    # Pricing
    PRICING_READ = "pricing:read"
    PRICING_CONFIGURE = "pricing:configure"
    PRICING_EXPERIMENT = "pricing:experiment"

    # Demand Forecast
    FORECAST_READ = "forecast:read"
    FORECAST_TRAIN = "forecast:train"

    # Analytics
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"

    # Admin
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_AUDIT = "admin:audit"


class Role(str, Enum):
    """
    Predefined roles with associated permissions.
    """
    # Super admin - full access
    ADMIN = "admin"

    # Operations team - manage dispatch and monitoring
    OPERATOR = "operator"

    # Restaurant manager - manage their restaurant
    RESTAURANT_MANAGER = "restaurant_manager"

    # Courier - limited access to their orders
    COURIER = "courier"

    # Customer - order and track
    CUSTOMER = "customer"

    # API client - programmatic access
    API_CLIENT = "api_client"

    # Read-only observer
    OBSERVER = "observer"


# Role to permissions mapping
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions

    Role.OPERATOR: {
        Permission.ORDER_READ,
        Permission.ORDER_UPDATE,
        Permission.ORDER_ASSIGN,
        Permission.ORDER_CANCEL,
        Permission.COURIER_READ,
        Permission.COURIER_LOCATION_READ,
        Permission.RESTAURANT_READ,
        Permission.DISPATCH_RUN,
        Permission.DISPATCH_CONFIGURE,
        Permission.PRICING_READ,
        Permission.FORECAST_READ,
        Permission.ANALYTICS_READ,
    },

    Role.RESTAURANT_MANAGER: {
        Permission.ORDER_READ,
        Permission.ORDER_UPDATE,
        Permission.RESTAURANT_READ,
        Permission.RESTAURANT_UPDATE,
        Permission.ANALYTICS_READ,
    },

    Role.COURIER: {
        Permission.ORDER_READ,
        Permission.ORDER_UPDATE,
        Permission.COURIER_READ,
        Permission.COURIER_UPDATE,
    },

    Role.CUSTOMER: {
        Permission.ORDER_CREATE,
        Permission.ORDER_READ,
        Permission.ORDER_CANCEL,
        Permission.RESTAURANT_READ,
        Permission.PRICING_READ,
    },

    Role.API_CLIENT: {
        Permission.ORDER_CREATE,
        Permission.ORDER_READ,
        Permission.COURIER_READ,
        Permission.RESTAURANT_READ,
        Permission.PRICING_READ,
        Permission.FORECAST_READ,
    },

    Role.OBSERVER: {
        Permission.ORDER_READ,
        Permission.COURIER_READ,
        Permission.RESTAURANT_READ,
        Permission.ANALYTICS_READ,
    },
}


@dataclass
class AuthorizationContext:
    """Context for authorization decisions."""
    user: TokenPayload
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None

    # Additional context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuthorizationService:
    """
    Authorization service implementing RBAC.

    Provides methods to check permissions and roles.
    """

    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS.copy()

    def get_user_permissions(self, user: TokenPayload) -> set[Permission]:
        """Get all permissions for a user based on their roles."""
        permissions = set()

        # Add permissions from roles
        for role_name in user.roles:
            try:
                role = Role(role_name)
                permissions.update(self.role_permissions.get(role, set()))
            except ValueError:
                logger.warning(f"Unknown role: {role_name}")

        # Add explicit permissions
        for perm_name in user.permissions:
            try:
                permissions.add(Permission(perm_name))
            except ValueError:
                logger.warning(f"Unknown permission: {perm_name}")

        return permissions

    def has_permission(
        self,
        user: TokenPayload,
        permission: Permission,
    ) -> bool:
        """Check if user has a specific permission."""
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions

    def has_any_permission(
        self,
        user: TokenPayload,
        permissions: list[Permission],
    ) -> bool:
        """Check if user has any of the specified permissions."""
        user_permissions = self.get_user_permissions(user)
        return any(p in user_permissions for p in permissions)

    def has_all_permissions(
        self,
        user: TokenPayload,
        permissions: list[Permission],
    ) -> bool:
        """Check if user has all of the specified permissions."""
        user_permissions = self.get_user_permissions(user)
        return all(p in user_permissions for p in permissions)

    def has_role(
        self,
        user: TokenPayload,
        role: Role,
    ) -> bool:
        """Check if user has a specific role."""
        return role.value in user.roles

    def has_any_role(
        self,
        user: TokenPayload,
        roles: list[Role],
    ) -> bool:
        """Check if user has any of the specified roles."""
        return any(r.value in user.roles for r in roles)

    def authorize(
        self,
        context: AuthorizationContext,
        required_permission: Permission,
    ) -> bool:
        """
        Authorize an action with full context.

        This is the main authorization entry point.
        """
        # Check permission
        if not self.has_permission(context.user, required_permission):
            logger.warning(
                "Authorization denied",
                user_id=context.user.sub,
                permission=required_permission.value,
                resource_type=context.resource_type,
                resource_id=context.resource_id,
            )
            return False

        # Additional resource-level checks could go here
        # e.g., check if user owns the resource

        logger.debug(
            "Authorization granted",
            user_id=context.user.sub,
            permission=required_permission.value,
        )
        return True


# Global service instance
_auth_service = AuthorizationService()


def require_permission(permission: Permission):
    """
    FastAPI dependency to require a specific permission.

    Usage:
        @router.get("/admin")
        async def admin_route(
            user: TokenPayload = Depends(require_permission(Permission.ADMIN_SYSTEM))
        ):
            return {"status": "ok"}
    """
    async def permission_checker(
        user: TokenPayload = Depends(get_current_user),
    ) -> TokenPayload:
        if not _auth_service.has_permission(user, permission):
            logger.warning(
                f"Permission denied",
                user_id=user.sub,
                required=permission.value,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission.value} required",
            )
        return user

    return permission_checker


def require_any_permission(*permissions: Permission):
    """Require any of the specified permissions."""
    async def permission_checker(
        user: TokenPayload = Depends(get_current_user),
    ) -> TokenPayload:
        if not _auth_service.has_any_permission(user, list(permissions)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions",
            )
        return user

    return permission_checker


def require_role(role: Role):
    """
    FastAPI dependency to require a specific role.

    Usage:
        @router.get("/operators-only")
        async def operator_route(
            user: TokenPayload = Depends(require_role(Role.OPERATOR))
        ):
            return {"status": "ok"}
    """
    async def role_checker(
        user: TokenPayload = Depends(get_current_user),
    ) -> TokenPayload:
        if not _auth_service.has_role(user, role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role.value}",
            )
        return user

    return role_checker


def require_any_role(*roles: Role):
    """Require any of the specified roles."""
    async def role_checker(
        user: TokenPayload = Depends(get_current_user),
    ) -> TokenPayload:
        if not _auth_service.has_any_role(user, list(roles)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role",
            )
        return user

    return role_checker


class ResourceAuthorizer:
    """
    Resource-level authorization.

    Checks if user can access a specific resource instance.
    """

    @staticmethod
    async def can_access_order(
        user: TokenPayload,
        order_id: str,
        get_order_func: Callable,
    ) -> bool:
        """Check if user can access a specific order."""
        # Admins and operators can access all orders
        if _auth_service.has_any_role(user, [Role.ADMIN, Role.OPERATOR]):
            return True

        # Get order
        order = await get_order_func(order_id)
        if order is None:
            return False

        # Customers can only access their own orders
        if _auth_service.has_role(user, Role.CUSTOMER):
            return str(order.customer_id) == user.sub

        # Couriers can access assigned orders
        if _auth_service.has_role(user, Role.COURIER):
            return str(order.courier_id) == user.sub

        # Restaurant managers can access their restaurant's orders
        if _auth_service.has_role(user, Role.RESTAURANT_MANAGER):
            # Would need to check restaurant ownership
            pass

        return False

    @staticmethod
    async def can_access_courier(
        user: TokenPayload,
        courier_id: str,
    ) -> bool:
        """Check if user can access courier data."""
        # Admins and operators can access all couriers
        if _auth_service.has_any_role(user, [Role.ADMIN, Role.OPERATOR]):
            return True

        # Couriers can only access their own data
        if _auth_service.has_role(user, Role.COURIER):
            return courier_id == user.sub

        return False

