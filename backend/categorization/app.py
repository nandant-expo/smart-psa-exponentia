import logging
import json
from typing import Union
from bson import json_util
from pydantic import BaseModel,Field
from fastapi import APIRouter, Response, BackgroundTasks,Security
# from common.chat_utils import index_one_inrecord
from common.categorization_utils import (process_categories,
                          generate_categorization_summary)
from fastapi_jwt import JwtAuthorizationCredentials
from common.jwt import access_security

router = APIRouter()

class CategorizationModule(BaseModel):
    # id : dict = Field(None, alias='_id')
    # display_name: Union[str, None] = None
    upload_path : Union[str, None] = None
    # static_path: Union[str, None] = None

@router.post("/categorise_file")
def categorize_file(excel: CategorizationModule, response: Response, credentials: JwtAuthorizationCredentials = Security(access_security)):
    try:
        workspace_name=credentials.subject['workspace']
        status, result = process_categories(excel.upload_path)
        # print(result)
        if status: 
            summary_status, summary_result = generate_categorization_summary(result)
            if summary_status:
                response.status_code = 200
                return {
                    "columns": [
                        {"field":"Plant Customer","header":"Plant Customer"},
                        {"field":"Description","header":"Description"},
                        {"field":"Product Name","header":"Product Name"},
                        {"field":"Machine Type","header":"Machine Type"},
                        {"field":"Length","header":"Length"},
                        {"field":"Height","header":"Height"},
                        {"field":"Width","header":"Width"},
                        {"field":"Weight","header":"Weight"},
                        {"field":"Category","header":"Product Type"},
                        {"field":"Confidence_Score","header":"Confidence Score"}],
                    "table" : json.loads(json_util.dumps(result)),
                    "summary" : summary_result
                    }
            else:
                response.status_code = 200
                return {
                    "columns": [
                        {"field":"Plant Customer","header":"Plant Customer"},
                        {"field":"Description","header":"Description"},
                        {"field":"Product Name","header":"Product Name"},
                        {"field":"Machine Type","header":"Machine Type"},
                        {"field":"Length","header":"Length"},
                        {"field":"Height","header":"Height"},
                        {"field":"Width","header":"Width"},
                        {"field":"Weight","header":"Weight"},
                        {"field":"Category","header":"Category"},
                        {"field":"Confidence_Score","header":"Confidence Score"}],
                    "table" : json.loads(json_util.dumps(result)),
                    "summary" : ""
                    }
        response.status_code = 409
        return {
            "message": result
            }
    except Exception as error:
        logging.error(f'Failed to extract PPT data. Error: {error}')
        response.status_code = 500
        return {
            "message": "Failed to extract PPT and insert ppt data."
            }