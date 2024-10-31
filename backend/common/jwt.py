from datetime import timedelta
import os
from fastapi_jwt import (
    JwtAccessBearer,
    JwtAuthorizationCredentials,
    JwtRefreshBearer,
)
from common.keyvault_connection import get_conn

client=get_conn()
access_security = JwtAccessBearer(
    secret_key=client.get_secret('jwtSecret').value,
    auto_error=True,
    access_expires_delta=timedelta(days=365)
)

refresh_security = JwtRefreshBearer(
    secret_key=client.get_secret('JWT-REFRESH-SECRET-KEY').value, 
    auto_error=True,
    refresh_expires_delta=timedelta(days=366)
)