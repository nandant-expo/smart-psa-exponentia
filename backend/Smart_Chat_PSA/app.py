from fastapi import FastAPI, HTTPException
from fastapi import APIRouter,Response,Security
from pydantic import BaseModel
from databricks.sql import connect
from datetime import datetime
import json
from fastapi_jwt import JwtAuthorizationCredentials
from common.jwt import access_security

router = APIRouter()

class Invoice(BaseModel):
    invoice_id : str 

# Establish connection to Databricks
def get_connection():
    return connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_ACCESS_TOKEN
    )


@router.get("/fetch_invoice")
async def fetch_invoices(credentials: JwtAuthorizationCredentials = Security(access_security)):
    query = "SELECT * FROM PSA_FINAL_TABLE"
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                records = cursor.fetchall()
                total_invoices = len(records)
                non_compliance_count = sum(1 for record in records if record[8] is False)
                updated_till = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                invoices = []
                for record in records:
                    discrepancies = record[7]
                    try:
                        # Attempt to parse the discrepancies field
                        discrepancies = json.loads(discrepancies)
                    except json.JSONDecodeError:
                        # If parsing fails, keep it as is
                        discrepancies = record[7]
                    
                    invoices.append({
                        "invoice_id": record[0],
                        "invoice_title": record[2],
                        "invoice_date": record[3],
                        "is_resolved": record[8],
                        "discrepancies": discrepancies
                    })

                return {
                    "message": "Invoices fetched successfully",
                    "data": {
                        "total_invoices": total_invoices,
                        "non_complying_invoices": non_compliance_count,
                        "updated_till": updated_till,
                        "invoices": invoices
                    }
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fetch_comparison/{invoice_id}")
async def fetch_comparisons(invoice_id: str,credentials: JwtAuthorizationCredentials = Security(access_security)):
    query = f"SELECT comparison FROM PSA_FINAL_TABLE WHERE invoice_no = '{invoice_id}'"  
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                record = cursor.fetchone() 
                
                if record:
                    return {
                        "message": "Comparison fetched successfully",
                        "data": {
                            "invoice_id": invoice_id,
                            "comparison": record[0] 
                        }
                    }
                else:
                    raise HTTPException(status_code=404, detail="Invoice not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.delete("/delete_invoice/{invoice_id}")
async def delete_invoice(invoice_id: str, credentials: JwtAuthorizationCredentials = Security(access_security)):
    query = f"DELETE FROM PSA_FINAL_TABLE WHERE invoice_no = '{invoice_id}'"  
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                if cursor.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Invoice not found")
                
                return {
                    "message": "Invoice deleted successfully",
                    }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/invoices_status/{invoice_id}")
async def resolve_invoice(invoice_id: str, credentials: JwtAuthorizationCredentials = Security(access_security)):
    query = f"UPDATE PSA_FINAL_TABLE SET resolved = True WHERE invoice_no = '{invoice_id}'"
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                if cursor.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Invoice not found")
                
                return {
                    "message": "Invoice resolved successfully"
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))