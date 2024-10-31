from fastapi import APIRouter,Security,Response
from fastapi_jwt import JwtAuthorizationCredentials
from common.jwt import access_security
from common.database import Database
from common.get_site_data import fetch_table_data
from common.PSA_analyticsChat_utils import (specific_flow_process,
                                        retrieve_chats,
                                        toggle_feedback,
                                        delete_chat,
                                        fetch_chat_history,
                                        retrieve_chat,generate_stream_response,
                                        check_graph_possibility,create_visualisation,analyse_data)
import json
from bson import json_util
from pydantic import BaseModel
import logging
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder


router=APIRouter()

@router.get("/fetch_table_psa")
def fetch_table_psa(response:Response,credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        db=Database('workspace','central')
        query=[{'$match': {'workspace_name': credentials.subject['workspace'],"catalog_name":{"$not":{"$eq":None}}, "schema_name":{"$not":{"$eq":None}}}},{"$project":{"_id":0,"host":1,"token":1,"catalog_name":1,"schema_name":1}}]
        status_connector,result_connector=db.fetch_aggregate(query)
        if status_connector:
            result_connector=result_connector[0]
            status,data=fetch_table_data(result_connector['host'],result_connector['token'],result_connector['catalog_name'],result_connector['schema_name'])
            if status:
                response.status_code=200
                return data
            else:
                response.status_code=200
                return data
        else:
            response.status_code=200
            return {"status":False,"data":[],"message":"Schema and Catalog information is not available"}
    except Exception as e:
        response.status_code=500
        return {"status":False,"message":repr(e)}

class AnalyticsChatModule(BaseModel):
    chat_id : dict = None
    user_id : dict = None
    table_id : str = None
    query : str = None
    url : str = None
    display_name : str = None
    feedback : int = None
    index : int = None

@router.post("/table_process_psa")
def fetch_table_psa(user:AnalyticsChatModule,response:Response,credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        user_id=credentials.subject['username']
        status,result = specific_flow_process(user.table_id, user_id,workspace_name)
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

@router.post("/table_chats_psa")
def get_chat_psa(user : AnalyticsChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
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

@router.post("/table_feedback_toggle_psa")
def feedback_toggle_psa(user : AnalyticsChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        status,result = toggle_feedback(json_util.loads(json.dumps(user.chat_id)), int(user.feedback),int(user.index),workspace_name)
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
    
@router.post("/table_chat_delete_psa")
def chat_delete_psa(user : AnalyticsChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
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

@router.post("/table_chat_history_psa")
def get_chat_history_psa(user : AnalyticsChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
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

@router.post("/table_stream_chat_psa")
def chat_psa(user : AnalyticsChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
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
    
@router.post("/table_get_chat_psa")
def get_chat_psa(user : AnalyticsChatModule, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
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

class AnalyticsChatModuleMultiTable(BaseModel):
    query: str=None
    chat_id : dict = None
    type: str=None
    sqlQuery: str = None
    file_location: str= None
    graph_type: str = None


@router.post("/decision_making_psa")
def get_chat_psa(user : AnalyticsChatModuleMultiTable, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        if user.chat_id != None:
            chat_id=json_util.loads(json.dumps(user.chat_id))
        else:
            chat_id=user.chat_id
        workspace_name=credentials.subject['workspace']
        status,result = check_graph_possibility(user_query=user.query,workspace_name=workspace_name,user_id=credentials.subject['username'],chat_id=chat_id)
        if status:
            logging.info('Checked Possible Response')
            response.status_code = 200            
            return json.loads(json_util.dumps(result))
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to Generate SQL query.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to Generate SQL query"                
                }
    
@router.post("/visualisation_psa")
def get_chat_psa(user : AnalyticsChatModuleMultiTable, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        if user.chat_id != None:
            chat_id=json_util.loads(json.dumps(user.chat_id))
        else:
            chat_id=user.chat_id
        workspace_name=credentials.subject['workspace']
        status,result = create_visualisation(user_id=credentials.subject['username'],workspace_name=workspace_name,chat_id=chat_id,file_location=user.file_location,graph_type=user.graph_type)
        if status:
            logging.info('Visualisation Creation')
            response.status_code = 200            
            return json.loads(json_util.dumps(result))
        response.status_code = 500        
        return {
                "message" : result                
                }
    except Exception as error:
        logging.error(f'Failed to create visualisation.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to create visualisation."                
                }

@router.post("/data_analysis_psa")
def get_chat_psa(user : AnalyticsChatModuleMultiTable, response : Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        if user.chat_id != None:
            chat_id=json_util.loads(json.dumps(user.chat_id))
        else:
            chat_id=user.chat_id
        workspace_name=credentials.subject['workspace']
        def generate_stream_response():

            for status,result in analyse_data(user_id=credentials.subject['username'],workspace_name=workspace_name,chat_id=chat_id,file_location=user.file_location,graph_type=user.graph_type,user_query=user.query,sql_query=user.sqlQuery):
                response.status_code = 200
                if status:
                    if status == 'stream':
                        yield f"data: {result}\n\n".encode()
                    else:
                        yield f"data: {result}\n\n".encode()
                else:
                    response.status_code = 400 
                    yield f"data: {jsonable_encoder({'message': result})}\n\n"
        return StreamingResponse(generate_stream_response(), media_type="text/event-stream")
        
    except Exception as error:
        logging.error(f'Failed to create visualisation.Error: {error}')
        response.status_code = 500        
        return {
                "message" : "Failed to create visualisation."                
                }