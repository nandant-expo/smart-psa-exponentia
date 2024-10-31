import json
import logging
from pymongo import MongoClient
import os
from bson import json_util
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Union,Optional
from fastapi import FastAPI, Response,Security
from common.utils import sync_file_access,check_credentials
from common.jwt import access_security,refresh_security
from fastapi_jwt import JwtAuthorizationCredentials


router = APIRouter()




class LoginModule(BaseModel):
    username : Union[str, None] = None    
    password : Optional[str] = None
    user_id : Optional[dict] = None
    access_token : Optional[str] = None
    loginType: Optional[str] = None
    personal_access_token: str = None

@router.post("/login")
async def login(user : LoginModule, response : Response):
    try:
        username = user.username        
        password = user.password 
        loginType=user.loginType       
        status,result = check_credentials(username,password,loginType,'central')
        if status:
            data={"role" : result[0],"user_id" : json.loads(result[1]),"username" : result[2],"displayname":result[3],"workspace":result[5]}
            access_token=access_security.create_access_token(subject=data)
            refresh_token=refresh_security.create_refresh_token(subject=data)
            logging.info('Successfully authenticated user.')
            response.status_code = 200            
            return {
                "status": result[4],
                "role" : result[0],
                "user_id" : json.loads(result[1]),
                "username" : result[2],
                "displayname":result[3],  
                "access_token":access_token,
                "refresh_token":refresh_token       
                }
        response.status_code = 401        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to authenticate.Error: {error}')
        response.status_code = 401        
        return {
                "message" : "Failed to authenticate user"                
                }

@router.post("/access_sync")
async def sync_access(user : LoginModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace=credentials.subject['workspace']  
        username = credentials.subject['username']     
        status,result = sync_file_access(username,user.personal_access_token,workspace)
        if status:
            logging.info('Successfully sync file access.')
            response.status_code = 200            
            return {
                "message" : result           
                }
        response.status_code = 401        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to sync file access.Error: {error}')
        response.status_code = 401        
        return {
                "message" : "Failed to sync file access."                
                }