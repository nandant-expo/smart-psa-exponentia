
import requests
from functools import partial
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool
from functools import reduce
from databricks.sdk import WorkspaceClient
from common.database import Database
import time
from databricks.connect import DatabricksSession

base_url="https://graph.microsoft.com/v1.0/sites"

def generate_token():
    url = f"https://login.microsoftonline.com/{tenant_id_1}/oauth2/token"

    payload=f'grant_type=client_credentials&client_id={client_id_1}&client_secret={client_secret_1}&resource=https%3A%2F%2Fgraph.microsoft.com%2F'
    headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Cookie': 'fpc=; stsservicecookie=; x-ms-gateway-slice='
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    if response.status_code==200:
        return True,response.json()['access_token']
    if response.status_code==400 or response.status_code==401:
        return False,"Invalid Client"

def list_data(access_token,headers,url,type):
    docs=[]
    headers['Authorization']=f'Bearer {access_token}'
    response = requests.request("GET", url, headers=headers, data={})
    data = response.json()
    if response.status_code==200:
        docs=docs+data['value']
        if type=='folder':docs = list(map(lambda item: {**item, "name": item['webUrl'].replace('%20',' ').split(f'sites/',1)[1].split('/',1)[1]} if "webUrl" in item else item, docs))
        if type=='list':docs=list(filter(lambda x:[x['list']['template']=='documentLibrary',x.pop('list')][0],docs))
        if '@odata.nextLink' in data:
            docs=docs+list_data(access_token,headers,data['@odata.nextLink'],type)
    if response.status_code==401:
        status,access_token=generate_token()
        docs=docs+list_data(access_token,headers,url,type)
        
    return docs

def get_folders(access_token,id,documents_list):
    folder_list=list_data(access_token,headers={'Prefer': 'HonorNonIndexedQueriesWarningMayFailRandomly','Accept': 'application/json;odata.metadata=none'},url=f"{base_url}/{id}/lists/{documents_list['id']}/items?$filter=fields/DocIcon eq null and fields/ContentType eq 'Folder'&$select=id,webUrl",type='folder')
    print(folder_list)
    return folder_list

def get_documents(access_token,list_sites):
    
    id=list_sites["id"]
    name=list_sites['displayName']
    documents_list=list_data(access_token,headers={'Accept': 'application/json;odata.metadata=none'},url=f"{base_url}/{id}/lists?$select=id,name,webUrl,list",type='list')
    try:
        overall_folders=documents_list
        if len(documents_list):
            reboot_pool = ThreadPool()
            func1=partial(get_folders,access_token,id)
            
            power_down = reboot_pool.map_async(func1,documents_list,chunksize=None)

            overall_folders=overall_folders+reduce(lambda x,y :x+y ,power_down.get())
            reboot_pool.terminate()
        final_output={"id":id,"name":name,"folders":overall_folders}
        
        return final_output
    except Exception as Error:
        reboot_pool.terminate()
        return Error

def get_site_folders(client_id,tenant_id,client_secret):
    try:
        global client_id_1;global tenant_id_1;global client_secret_1
        client_id_1=client_id;tenant_id_1=tenant_id;client_secret_1=client_secret
        status,access_token=generate_token()
        if status==True:
            list_site=list_data(access_token=access_token,headers={},url = f"{base_url}/getAllSites?$filter=isPersonalSite eq false and displayName ne null&$select=id,displayName,isPersonalSite",type='site')
            if len(list_site):
                pool = Pool(processes=8) 
                func = partial(get_documents,access_token)
                p1=(pool.map(func,list_site,chunksize=None))
                pool.close()
                return True,p1
            else:
                return True,[]
        else:
            return False,access_token
    except Exception as e:
        print(e)
        pool.close()
        return False,"Something went wrong"


def get_schemas_tables(host,token,item):
    
    api_client=WorkspaceClient(host=host,token=token)
    schemas=list(filter(None,map(lambda schema: {"schema_name":schema.name} if schema.name!="information_schema" else None, api_client.schemas.list(catalog_name=item['name']))))
    
    data={"catalog_name":item['name'],"catalog_data":schemas}
    return data

def get_databricks_data(host,token):
    try:

        api_url = f"{host}/api/2.1/unity-catalog/catalogs"
        headers = {"Authorization": f"Bearer {token}"}

        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 200:
            catalog_list = response.json()
            catalog_list=catalog_list['catalogs']
            pool = Pool(processes=3) 
            func = partial(get_schemas_tables,host,token)
            p1=pool.map(func,catalog_list,chunksize=None)

            api_client=WorkspaceClient(host=host,token=token)

            data=list(map(lambda cluster: {"cluster_name":cluster.cluster_name,"cluster_id":cluster.cluster_id}, api_client.clusters.list()))
            final_data={"status":True,"catalog_information":p1,"cluster_information":data}
            return True,final_data
        else:
            final_data={"status":False,"catalog_information":[],"cluster_information":[]}
            final_data={**final_data, **response.json()}
            return False,final_data
    except requests.exceptions.RequestException:
        return False,{"status":False,"catalog_information":[],"cluster_information":[],"message":"Invalid Host"}
    except Exception as e:
        return False,{"status":False,"catalog_information":[],"cluster_information":[],"message":repr(e)}

def fetch_table_data(host,token,catalog_name,schema_name):
    try:
        api_client=WorkspaceClient(host=host,token=token)
        tables=list(map(lambda table: {"_id":table.full_name,"table_name":table.name}, api_client.tables.list(catalog_name=catalog_name,schema_name=schema_name)))
        return True,{"status":True,"data":tables}
    except Exception as e:
        return False,{"status":False,"data":[],"message":repr(e)}
    

def query_to_dataframe(host,token,query_result,catalog,schema,cluster_id):
    start=time.time()
    spark = DatabricksSession.Builder().remote(host=host,token=token,cluster_id=cluster_id).getOrCreate()

    spark.sql(f"use catalog {catalog}")
    spark.sql(f"use schema {schema}")
    df = spark.sql(f"""{query_result}""")
    df=df.toPandas()
    spark.stop()
    end=time.time()
    print(end-start)
    return df

def     save_metadata_information(data,workspace_name):
    try:
        api_client = WorkspaceClient(host=data['host'], token=data['token'])
        tables = api_client.tables.list(catalog_name=data['catalog_name'], schema_name=data['schema_name'])

        for table in tables:
            table_full_name=table.full_name
            table_name=table.name
            df=query_to_dataframe(data['host'],data['token'],f"select * from {table_full_name} limit 1",data['catalog_name'],data['schema_name'],data['cluster_name'])
            # df=df.fillna(None)
            dict_of_df = df.to_dict()
            constraints = api_client.tables.get(table.full_name).as_dict()
            print(constraints)
            metadata={"catalog_name":data['catalog_name'],"schema_name":data['schema_name'],"table_name":table.name,"columns":list(map(lambda schema: {"name":schema['name'],"datatype":schema['type_name'],"Column Description": schema['comment'] if 'comment' in schema else None ,"sample_data":str(dict_of_df[schema['name']][0])},constraints["columns"]))}
            

            if "table_constraints" in constraints:
                metadata['table_constraint']=constraints["table_constraints"]
            # print(metadata)
            db=Database('tebleMetadata',workspace_name)
            status_insert,result_insert=db.insert_single_record(metadata)
            
        
        return True
    except Exception as error:
        print(error)
        return False