from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from src.api.auth import decode_access_token
from src.api.schemas import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
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
                detail="Operation not permitted: Insufficient permissions"
            )
        return current_user
    return role_checker