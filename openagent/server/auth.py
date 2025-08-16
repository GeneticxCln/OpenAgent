"""
Authentication manager for OpenAgent server.

Handles JWT token generation, validation, and user authentication
with configurable security settings.
"""

import os
try:
    import jwt  # type: ignore
except ImportError:  # pragma: no cover
    jwt = None  # Fallback: disable JWT features when library is unavailable
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .models import User, LoginResponse


class AuthConfig:
    """Authentication configuration."""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY") or secrets.token_urlsafe(32)
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
        self.auth_enabled = os.getenv("AUTH_ENABLED", "false").lower() == "true"


class AuthManager:
    """Authentication manager for handling JWT tokens and user authentication."""
    
    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self.security = HTTPBearer(auto_error=False)
        
        # Simple in-memory user store (replace with real database in production)
        self._users: Dict[str, Dict[str, Any]] = {
            "admin": {
                "id": "admin",
                "username": "admin",
                "email": "admin@example.com",
                "password_hash": self._hash_password("admin123"),
                "is_active": True,
                "created_at": datetime.now(timezone.utc),
                "last_login": None,
                "roles": ["admin"]
            },
            "user": {
                "id": "user",
                "username": "user",
                "email": "user@example.com",
                "password_hash": self._hash_password("user123"),
                "is_active": True,
                "created_at": datetime.now(timezone.utc),
                "last_login": None,
                "roles": ["user"]
            }
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash a password with salt."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, stored_hash = password_hash.split(':')
            password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return stored_hash == password_hash_check.hex()
        except ValueError:
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password."""
        user_data = self._users.get(username)
        if not user_data or not user_data["is_active"]:
            return None
        
        if not self._verify_password(password, user_data["password_hash"]):
            return None
        
        # Update last login
        user_data["last_login"] = datetime.now(timezone.utc)
        
        return User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            is_active=user_data["is_active"],
            created_at=user_data["created_at"],
            last_login=user_data["last_login"],
            roles=user_data.get("roles", [])
        )
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=self.config.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        if jwt is None:  # pragma: no cover
            # Minimal fallback token representation when PyJWT is not installed
            return "noop-token"
        encoded_jwt = jwt.encode(
            to_encode,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        if jwt is None:  # pragma: no cover
            return None
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm]
            )
            return payload
        except Exception:
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by their ID."""
        for user_data in self._users.values():
            if user_data["id"] == user_id:
                return User(
                    id=user_data["id"],
                    username=user_data["username"],
                    email=user_data["email"],
                    is_active=user_data["is_active"],
                    created_at=user_data["created_at"],
                    last_login=user_data["last_login"],
                    roles=user_data.get("roles", [])
                )
        return None
    
    def login(self, username: str, password: str) -> LoginResponse:
        """Login a user and return an access token."""
        user = self.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token = self.create_access_token(data={"sub": user.id})
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=user,
            expires_in=self.config.access_token_expire_minutes * 60
        )
    
    async def get_current_user(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
    ) -> Optional[User]:
        """Get the current authenticated user from the request."""
        if not self.config.auth_enabled:
            # Return a default user when authentication is disabled
            return User(
                id="anonymous",
                username="anonymous",
                email=None,
                is_active=True,
                created_at=datetime.now(timezone.utc),
                last_login=None,
                roles=[]
            )
        
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        payload = self.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = self.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
    
    def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None
    ) -> User:
        """Create a new user."""
        if username in self._users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        user_id = secrets.token_urlsafe(16)
        password_hash = self._hash_password(password)
        user_data = {
            "id": user_id,
            "username": username,
            "email": email or f"{username}@example.com",
            "password_hash": password_hash,
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
            "last_login": None,
        }
        self._users[username] = user_data
        return User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            is_active=user_data["is_active"],
            created_at=user_data["created_at"],
            last_login=user_data["last_login"],
            roles=user_data.get("roles", [])
        )
        user_data = {
            "id": user_id,
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "is_active": True,
"created_at": datetime.now(timezone.utc),
            "last_login": None
        }
        
        self._users[username] = user_data
        
        return User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            is_active=user_data["is_active"],
            created_at=user_data["created_at"],
            last_login=user_data["last_login"]
        )
    
    def disable_user(self, username: str) -> bool:
        """Disable a user account."""
        if username in self._users:
            self._users[username]["is_active"] = False
            return True
        return False
    
    def enable_user(self, username: str) -> bool:
        """Enable a user account."""
        if username in self._users:
            self._users[username]["is_active"] = True
            return True
        return False
    
    def change_password(self, username: str, new_password: str) -> bool:
        """Change a user's password."""
        if username in self._users:
            self._users[username]["password_hash"] = self._hash_password(new_password)
            return True
        return False
    
    def list_users(self) -> list[User]:
        """List all users."""
        users = []
        for user_data in self._users.values():
            users.append(User(
                id=user_data["id"],
                username=user_data["username"],
                email=user_data["email"],
                is_active=user_data["is_active"],
                created_at=user_data["created_at"],
                last_login=user_data["last_login"]
            ))
        return users
