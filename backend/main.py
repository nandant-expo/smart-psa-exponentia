from fastapi import FastAPI,Request,Depends, HTTPException
from common.utils import clean_up, clean_query_cache
from chat.app import router as chat
from login.app import router as login
from upload.app import router as upload
from search.app import router as search
from validateUser.app import router as validateUser
from signup.app import router as signup
from users.app import router as users
from configurations.app import router as configurations
from connector_settings.app import router as connector_settings
from tenantDelete.app import router as tenantDelete
from application_settings.app import router as application_settings
from analytics_chat.app import router as analytics_chat
from categorization.app import router as categorization
from analytics_chat_PSA.app import router as analytics_chat_PSA
from wa_chat.app import router as wa_chat
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.gzip import GZipMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from Smart_Chat_PSA.app import router as Smart_Chat_PSA
import json


app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler(timezone='EST')
    scheduler.add_job(clean_up, 'cron', hour=0, minute=0)
    # scheduler.add_job(clean_query_cache, 'cron', hour=0, minute=0)
    scheduler.start()

@app.get("/")
def greetings():
    return "Welcome to Exponentia.ai's Search Engine Web Application\n Version 2.0.5"

app.include_router(login, tags=['Login Module'])
app.include_router(upload, tags=['Upload Module'])
app.include_router(search, tags=['Search Module'])
app.include_router(chat, tags=['Chat Module'])
app.include_router(validateUser, tags=['Tenant Onboard Module'])
app.include_router(signup, tags=['Tenant Onboard Module'])
app.include_router(users,tags=['User Onboard Module'])
app.include_router(configurations,tags=['User Configuration'])
app.include_router(connector_settings,tags=['Connector Settings'])
app.include_router(tenantDelete,tags=["Tenant Delete"])
app.include_router(application_settings,tags=["Application Settings"])
app.include_router(analytics_chat,tags=["Analytics Chat Module"])
app.include_router(categorization,tags=["Categorization Module"])
app.include_router(analytics_chat_PSA,tags=['Analytics Chat PSA'])
app.include_router(wa_chat,tags=['WhatsApp chat module'])
app.include_router(Smart_Chat_PSA,tags=['Smart contract PSA module'])