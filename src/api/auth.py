import os
from datetime import datetime, timedelta
from typing import Optional
import yaml
from jose import JWTError, jwt
from passlib.context import CryptContext


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# تحميل الإعدادات
config = load_config()
auth_cfg = config.get("auth", {})
jwt_cfg = auth_cfg.get("jwt", {})

# إعداد المتغيرات من الـ Config والـ Environment
SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", 
    jwt_cfg.get("default_secret_key", "super-secret-key-change-this-in-production")
)
ALGORITHM = jwt_cfg.get("algorithm", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = jwt_cfg.get("access_token_expire_minutes", 1440)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None