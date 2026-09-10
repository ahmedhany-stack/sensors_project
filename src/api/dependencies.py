import os
import yaml
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from src.api.auth import decode_access_token
from src.api.schemas import TokenData


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# تحميل الإعدادات
config = load_config()
auth_cfg = config.get("auth", {})
auth_msgs = auth_cfg.get("messages", {})

# جلب tokenUrl من الـ Config مع قيمة افتراضية
TOKEN_URL = auth_cfg.get("token_url", "token")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=TOKEN_URL)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=auth_msgs.get("invalid_credentials", "Could not validate credentials"),
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    
    username: str = payload.get("sub")
    role: str = payload.get("role")
    
    if username is None or role is None:
        raise credentials_exception
        
    return TokenData(username=username, role=role)


def require_role(allowed_roles: list[str]):
    def role_checker(current_user: TokenData = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=auth_msgs.get("forbidden", "Operation not permitted: Insufficient permissions")
            )
        return current_user
    return role_checker