import json
from io import BytesIO
from pydantic import BaseModel
from common.database import Database
from common.jwt import access_security
from common.get_site_data import get_site_folders
from fastapi_jwt import JwtAuthorizationCredentials
from fastapi import APIRouter, Response, Security, UploadFile, Form, File, Request
from common.utils import test_user_application , data_to_cosmos, delete_app_pipeline, delete_application, application_output, application_output_external
from bson import json_util,ObjectId

class Application(BaseModel):
    application_id : dict = {}
    pipeline_id : dict = {}
    data : dict = {}


router=APIRouter()

def rename_key(data, old_key, new_key):
    if isinstance(data, dict):  # If data is a single dictionary
        if old_key in data:
            data[new_key] = data.pop(old_key)
    elif isinstance(data, list):  # If data is a list of dictionaries
        for item in data:
            if old_key in item:
                item[new_key] = item.pop(old_key)
    else:
        print("Unsupported data type. Function supports dictionaries and lists of dictionaries.")

@router.get('/configuartion')
def Configuartion(response:Response,credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        db_central=Database('workspace','central')
        db_workspace=Database('pipelineSettings',credentials.subject['workspace'])
        query=[{'$match': {'workspace_name': credentials.subject['workspace'],"clientID":{"$not":{"$eq":None}}, "tenantID":{"$not":{"$eq":None}}, "clientSecret":{"$not":{"$eq":None}}}},{"$project":{"_id":0,"clientID":1,"tenantID":1,"clientSecret":1}}]
        status_connector,result_connector=db_central.fetch_aggregate(query)
        dataSource=[{"key":"sharepoint","displayName":"Share Point"}]
        frequencyList=[{ "key":"hourly","displayName":"Hourly" },{ "key":"daily","displayName":"Daily" },{ "key":"weekly","displayName":"Weekly" }]
        inputFileFormats=[{ "key":"pdf","displayName":"Pdf" },{ "key":"docx","displayName":"Docx" },{ "key":"xlsx","displayName":"Xlsx" },{ "key":"pptx","displayName":"Pptx" },{ "key":"wav","displayName":"Wav" },{ "key":"mp3","displayName":"Mp3" },{ "key":"mp4","displayName":"Mp4" }]
        outputFileFormats=[{ "key":"csv","displayName":"CSV" },{ "key":"json","displayName":"JSON" }]
        if status_connector:
            result_connector=result_connector[0]
            status,data=get_site_folders(result_connector["clientID"],result_connector["tenantID"],result_connector["clientSecret"])
            if status:
                response.status_code=200
                return {"connectionFlag":True,"setingFlag":True,"message":"","siteData":data,"dataSource":dataSource,"frequencyList":frequencyList,"inputFileFormats":inputFileFormats,"outputFileFormats":outputFileFormats}
            if data=='Invalid Client':
                response.status_code=200
                return {"connectionFlag":False,"setingFlag":True,"message":data,"siteData":[],"dataSource":dataSource,"frequencyList":frequencyList,"inputFileFormats":inputFileFormats,"outputFileFormats":outputFileFormats}
            else:
                response.status_code=500
                return {"connectionFlag":False,"setingFlag":False,"message":"Somthing Went Wrong While Fetching Site folders"}
        if result_connector=='No records found.':
            response.status_code=200
            return {"connectionFlag":False,"setingFlag":False,"siteData":[],"dataSource":dataSource,"frequencyList":frequencyList,"inputFileFormats":inputFileFormats,"outputFileFormats":outputFileFormats}
        else:
            response.status_code=500
            return {"connectionFlag":False,"setingFlag":False,"message":"Something Went Wrong With Database Fetch"}
    except Exception as Error:
        response.status_code=500
        print(Error)
        return {"connectionFlag":False,"setingFlag":False,"message":"Something Went Wrong"}

@router.get('/applications')
def get_application_data(response:Response, request: Request,credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace = credentials.subject['workspace']
        db = Database('ApplicationCollection',workspace)
        status, res = db.fetch_all_records({},{'application_name':1,'created_at':1,'created_by':1,'modified_at':1,'modified_by':1})
        if status:
            rename_key(res, '_id', 'application_id')
            response.status_code=200
            # print([str(request.url_for('get_application_output',workspace=workspace,application_id= result['application_id'])) for result in res])
            [result.update({"output_api":str(request.url_for('get_application_output',workspace=workspace,application_id= result['application_id']))}) for result in res]
            data = json.loads(json_util.dumps(res))
            return {"data":data}
        else:
            response.status_code=200
            print(res)
            return {'data': []}
                
    except Exception as Error:
        response.status_code=500
        print(Error)
        return {"message":"Something Went Wrong"}

@router.get('/applications/{application_id}')
def get_application_data(application_id:str,response:Response,credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace = credentials.subject['workspace']
        db = Database('ApplicationCollection',workspace)
        status, res = db.fetch_one_record({'_id': ObjectId(application_id)})
        if status:
            rename_key(res, '_id', 'application_id')
            response.status_code=200
            res.pop('logs',None)
            data = json.loads(json_util.dumps(res))
            return data
        else:
            response.status_code=400
            return json.loads({'message': 'Failed to fetch application details.'})
    except Exception as Error:
        response.status_code=500
        print(Error)
        return {"message":"Something Went Wrong"}


@router.post('/applications/test')
async def test_instructions(response:Response, files: list[UploadFile] = File(), credentials: JwtAuthorizationCredentials = Security(access_security), test_data : str= Form()):
    try:
        workspace = credentials.subject['workspace']
        files = [
            {
                "file_name" : file.filename,
                "data": BytesIO(await file.read())
            } 
                  for file in files
        ]
        print("files: ",[file for file in files])
        test_data = json_util.loads(test_data)
        print("Files read successfully.")
        status,result = test_user_application(test_data,files,workspace)
        if status:
            response.status_code = 200
            return json.loads(json_util.dumps(result))
        else:
            response.status_code = 400
            print(f"Failed to test pipeline. {result}")
            return {'message':'Pipeline test failed'}
    except Exception as Error:
        response.status_code = 500
        print(Error)
        return {"message":"Something Went Wrong"}
    

@router.post('/applications')
def create_application(response:Response, application: Application, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace = credentials.subject['workspace']
        username = credentials.subject['displayname']
        application_data = json_util.loads(json.dumps(application.data))
        db_type = 'insert'
        print("type = insertion")
        status, id = data_to_cosmos(db_type, application_data, workspace, username)
        print("id: ", id)
        print("type of id: ", type(id))
        if status:
            response.status_code=200
            return {'application_id': json.loads(json_util.dumps(id)), 'message': "the data is inserted successfully in the database"}
        else:
            response.status_code=400
            return {'message': "the data is not inserted in the database"}
    except Exception as error:
        response.status_code=500
        
        print(error)
        return {"message": error}
    
@router.put('/applications/{application_id}')
def update_application(response:Response, application_id:str,  application: Application, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace = credentials.subject['workspace']
        username = credentials.subject['displayname']
        application_id = ObjectId(application_id)
        application_data = json_util.loads(json.dumps(application.data))
        application_data.update({
            'application_id':application_id
        })
        if application_data['application_id']:
            db_type = 'update'
            print("type = updation")
            status, msg = data_to_cosmos(db_type, application_data, workspace, username)
            if status:
                response.status_code=200
                return {"message": "successully updated application."}
            else:
                response.status_code=400
                return {"message": msg}
    except Exception as error:
        response.status_code=500
        print(error)
        return {"message": error}



@router.delete('/applications/{application_id}/pipelines/{pipeline_id}')
def delete_pipeline(response:Response, application_id:str, pipeline_id:str, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace = credentials.subject['workspace']
        application_id = ObjectId(application_id)
        pipeline_id = ObjectId(pipeline_id)
        status,message = delete_app_pipeline(application_id, pipeline_id, workspace)
        if status:
            response.status_code = 200
            return {"message" : message}
        else:
            response.status_code = 400
            return {"message" : message}
    except Exception as error:
        response.status_code=500
        print(error)
        return {"message": error}

@router.delete('/applications/{application_id}')
def delete_application_data(response:Response, application_id:str, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace = credentials.subject['workspace']
        application_id = ObjectId(application_id)
        status,message = delete_application(application_id, workspace)
        if status:
            response.status_code = 200
            return {"message" : message}
        else:
            response.status_code = 400
            return {"message" : message}
    except Exception as error:
        response.status = 500
        print('Failed to delete application')
        return {"message" : error}

@router.get('/applications/{application_id}/output-schema')
def get_application_output(response:Response, application_id:str,request: Request, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        app_base_url = request.url_for('greetings')
        workspace = credentials.subject['workspace']
        application_id = ObjectId(application_id)
        status,message = application_output(application_id, workspace, app_base_url)
        if status:
            response.status_code = 200
            return {"message" : "Successfully fetched output schema",
                    "data" : message}
        else:
            response.status_code = 400
            return {"message" : message}
    except Exception as error:
        response.status = 500
        print('Failed to delete application')
        return {"message" : error}


@router.get('/tenant/{workspace}/applications/{application_id}/output-schema')
def get_application_output(response:Response, workspace:str, application_id:str,request: Request):
    try:
        app_base_url = request.url_for('greetings')
        workspace = workspace
        application_id = ObjectId(application_id)
        status,message = application_output_external(application_id, workspace, app_base_url)
        if status:
            response.status_code = 200
            return {"message" : "Successfully fetched output schema",
                    "data" : message}
        else:
            response.status_code = 400
            return {"message" : message}
    except Exception as error:
        response.status = 500
        print('Failed to delete application')
        return {"message" : error}
