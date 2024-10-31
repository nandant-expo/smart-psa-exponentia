import json
from fastapi import APIRouter,Response,Security
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from bson import json_util
from common.chat_utils import (
    generate_stream_response,
    retrieve_chat,
    retrieve_chats,
    fetch_chat_history, 
    delete_chat,
    # index_all_inrecords,
    local_flow_process,
    local_flow_preview,
    local_index_update,
    toggle_feedback_v2,
    get_chat_preview,
    specific_flow_process,
    toggle_feedback
    )
import logging
from fastapi_jwt import JwtAuthorizationCredentials
from common.jwt import access_security

router = APIRouter()

class ChatModule(BaseModel):
    chat_id : dict = None
    user_id : dict = None
    file_id : dict = None
    query : str = None
    url : str = None
    display_name : str = None
    feedback : int = None
    index : int = None
    assessment : str = None
    feedback_description : str = None
    

@router.post("/stream_chat")
def chat(user : ChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        query = user.query
        user_id = credentials.subject['username']
        chat_id = json_util.loads(json.dumps(user.chat_id)) if user.chat_id else None

        def generate_stream():
            for status, result in generate_stream_response(query,user_id,chat_id,workspace=workspace_name):
                response.status_code = 200 
                if status:
                    if status == 'stream':
                        yield f"data: {result}\n\n".encode()
                    else:
                        yield f"data: {result}\n\n".encode()
                else:
                    response.status_code = 400 
                    yield f"data: {jsonable_encoder({'message': result})}\n\n"


        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    except Exception as error:
        logging.error(f'Failed to generate response.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to generate response."                
                }
    
@router.post("/get_chat")
def get_chat(user : ChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        status,result = retrieve_chat(json_util.loads(json.dumps(user.chat_id)),workspace_name)
        if status:
            logging.info('Successfully retrieve chat.')
            response.status_code = 200            
            return json.loads(json_util.dumps(result))
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to retrieve chat.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to retrieve chat ."                
                }
    
    
@router.post("/chats")
def get_chat(user : ChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        status,result = retrieve_chats(json_util.loads(json.dumps(user.chat_id)),workspace_name)
        if status:
            logging.info('Successfully retrieve chat.')
            response.status_code = 200            
            return json.loads(json_util.dumps(result))
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to retrieve chat.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to retrieve chat ."                
                }
    
@router.post("/chat_history")
def get_chat_history(user : ChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        user_id=credentials.subject['username']
        status,result = fetch_chat_history(user_id,workspace_name)
        if status:
            logging.info('Successfully retrieve chat history.')
            response.status_code = 200            
            return json.loads(json_util.dumps(result))
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to retrieve chat history.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to retrieve chat history."                
                }
    
@router.post("/chat_delete")
def chat_delete(user : ChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        status,result = delete_chat(json_util.loads(json.dumps(user.chat_id)),workspace_name)
        if status:
            logging.info('Successfully deleted chat record.')
            response.status_code = 200            
            return {'message':result}
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to delete chat record.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to delete chat record."                
                }
    
@router.post("/chat_process")
def chat_process(user : ChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        user_id=credentials.subject['username']
        status,result = local_flow_process(user.url,user.display_name,user_id,workspace_name)
        if status:
            logging.info('Successfully created chat record.')
            response.status_code = 200            
            return json.loads(json_util.dumps(result))
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to created chat record.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to created chat record."                
                }
    
@router.post("/chat_preview")
def chat_preview(user : ChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        status,result = local_flow_preview(user.url,workspace_name, user.display_name)
        if status:
            logging.info('Successfully created chat preview.')
            response.status_code = 200            
            return {"message":result}
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to created chat preview.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to created chat preview."                
                }
    
@router.post("/index_update")
def index_update(user : ChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        status,result = local_index_update(user.display_name,workspace_name)
        if status:
            logging.info('Successfully updated index.')
            response.status_code = 200            
            return {"message":result}
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to update index.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to update index."                
                }
    
@router.post("/get_preview")
def get_preview(user : ChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        status,result = get_chat_preview(json_util.loads(json.dumps(user.chat_id)),workspace_name)
        if status:
            logging.info('Successfully retrieved chat preview.')
            response.status_code = 200            
            return result
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to retrieve chat preview.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to retrieve chat preview."                
                }
    
@router.post("/chat_process2")
def chat_process(user : ChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        user_id=credentials.subject['username']
        status,result = specific_flow_process(json_util.loads(json.dumps(user.file_id)), user_id,workspace_name)
        if status:
            logging.info('Successfully created chat record.')
            response.status_code = 200            
            return json.loads(json_util.dumps(result))
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to created chat record.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to created chat record."                
                }

@router.post("/feedback_toggle")
def feedback_toggle(user : ChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        chat_id = json_util.loads(json.dumps(user.chat_id))
        feedback = int(user.feedback)
        index = int(user.index)
        status,result = toggle_feedback(chat_id, feedback, index, workspace_name)
        if status:
            logging.info('Successfully Updated Feedback.')
            response.status_code = 200            
            return {
                "message" : result                
                }
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to Update feedback.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to Update feedback."                
                }

@router.post("/feedback_toggle/v2")
def feedback_togglev2(user : ChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        chat_id = json_util.loads(json.dumps(user.chat_id))
        feedback = int(user.feedback)
        index = int(user.index)
        assessment = user.assessment
        feedback_description = user.feedback_description
        status,result = toggle_feedback_v2(chat_id, feedback, index, assessment, feedback_description, workspace_name)
        if status:
            logging.info('Successfully Updated Feedback.')
            response.status_code = 200            
            return {
                "message" : result                
                }
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to Update feedback.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to Update feedback."                
                }