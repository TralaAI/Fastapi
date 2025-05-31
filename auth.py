from fastapi import HTTPException, status
from fastapi.security import HTTPBearer
from jose import jwt, JWTError
from config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_ISSUER, JWT_AUDIENCE

security = HTTPBearer()

def verify_token(token: str) -> bool:
  try:
    if JWT_SECRET_KEY is None:
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="JWT secret key is not configured."
      )
    jwt.decode(
      token,
      JWT_SECRET_KEY,
      algorithms=[JWT_ALGORITHM] if JWT_ALGORITHM is not None else [],
      issuer=JWT_ISSUER,
      audience=JWT_AUDIENCE,
    )
    return True
  except JWTError:
    return False