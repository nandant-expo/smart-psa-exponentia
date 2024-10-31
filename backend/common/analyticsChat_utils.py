from common.database import Database
from common.logger import initiate_logger
from datetime import datetime
from common.keyvault_connection import get_conn
import openai
import tiktoken
import random
from retry import retry
import json
from bson import json_util
from databricks.connect import DatabricksSession
import pandas as pd
import time
from databricks.sdk import WorkspaceClient
from langchain_openai.chat_models.azure import AzureChatOpenAI
import threading
from azure.storage.blob import BlobServiceClient
from langchain_community.utilities import sql_database
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from databricks.connect import DatabricksSession
import base64
import os
from io import StringIO
import plotly
import re
from common.chat_utils import doc_search


client=get_conn()
LOGGER = initiate_logger('Analytics Chat Utils Logger')

openai.api_type = "azure"
openai.api_version = "2023-05-15"
GPT_TYPE = client.get_secret("GPT-TYPE").value



if GPT_TYPE == "GPT-4":
    openai_api_base = [client.get_secret("AZURE-OPENAI-ENDPOINT-SC").value,client.get_secret("AZURE-OPENAI-ENDPOINT-SC").value]
    openai_key = [client.get_secret('OPENAI-API-KEY-SC').value,client.get_secret('OPENAI-API-KEY-SC').value]
    
    DEPLOYMENT_NAME = [client.get_secret("SDA-MODEL").value, client.get_secret("SDA-MODEL").value]
else:
    openai_api_base = [client.get_secret("AZURE-OPENAI-ENDPOINT-US").value,client.get_secret("AZURE-OPENAI-ENDPOINT-FRANCE").value]
    openai_key = [client.get_secret('OPENAI-API-KEY-US').value,client.get_secret('OPENAI-API-KEY-FRANCE').value]
    DEPLOYMENT_NAME = ["chatgpt-16k", "chatgpt-16k"]

SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"
system_message_chat_conversation=""""
Welcome to AIXponent analytics module, Please provide your query, and I will assist you by referencing our extensive knowledge base.

"""

system_message_conversation="""Welcome to AIXponent analytics Chat:
    Give your answer by following below all 11 instructions:
    1. keep in mind these columns and table constraints information while giving answer but you should only use {table} table
    2. do not use table other then {table}
    3. no need to list tables as {table} table is available in the schema
    4. use only sql_db_query, not need to use sql_db_query_checker
    5. the structure of {table} table with columns and table constraints is {table_data}, construct query based on this data
    6. If data coming from user query is large and exceeds token limit then only fetch first 10 to 20 rows of that data
    7. Always use 'REGEXP' and also generate query which converts both column and value to upper case, for conditions in where cluase,if column data type is string.
    8. Only provide data which user asked, do not add extra information
    9. If user ask to describe table columns then get that information from {table_data}.
    10. There is a possibility of Final Answer, immediately after Action, So,if that situation occurs do not give error 
    11. if user query related information are not available in {table} table and stored in different table then give that table name as an answer"""

# global DB_DATABASE
DB_DATABASE =""

def specific_flow_process(id,user_id,workspace):#add userid
    status,chat_id = create_specific_chat(id,user_id,workspace)
    if status:
        db = Database('AnalyticsCollection',workspace)
        status,record = db.fetch_one_record({"_id":chat_id},{'title':1,'updated_on':1,'chat_flow':1})
        print(record)
        record['chat_id'] = record.pop('_id')
        return status,record
    return status, record

def create_specific_chat(id,user_id,workspace):
    try:
        now = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        db = Database('AnalyticsCollection',workspace)
        # llm_citations = False
        chat_meta = {}
        chat_conversations = [{"role" : SYSTEM, "content" : system_message_chat_conversation}]
        citation = [[]]
        analytics = [{
            "summary":"",
            "keywords":[],
            "sample_questions":[]
        }]
        feedback = [None]
        dates=[{"query_date": None,"feedback_date": None}]
        chat_conversations.append({"role" : ASSISTANT, "content" : ""})
        citation.append([])
        feedback.append(2)
        chat_meta['chat_flow'] = 'specific'
        analysis = get_analytics(id,chat_meta['chat_flow'],workspace)
        analytics.append(analysis)
        chat_meta['user_id'] = user_id
        chat_meta['created_on'] = now
        chat_meta['chat_conversations'] = chat_conversations
        chat_meta['analytics'] = analytics
        chat_meta['feedback'] = feedback
        chat_meta['dates'] = dates
        chat_meta['chat_table'] = id.split('.')[2]
        chat_meta['citations']=citation
        # print(id.split('.'))
        chat_meta['chat_table_id'] = id
        chat_meta["title"] = chat_title([analysis['summary']])
        print(chat_meta["title"])
        chat_meta['updated_on'] = now
        status,id = db.insert_single_record(chat_meta)
        message = id if not status else ""
        if not status:
            return False,message
        print('New Chat entry done.')
        return status,id
    except Exception as error:
        print(f'Failed chat_utils->create_local_chat. Error:{error}')
        return False,"Failed to create local chat"
def global_variable_declare(catalog,schema,host,token,cluster_id):
    host_name=host.removeprefix("https://")
    DB_DATABASE=sql_database.SQLDatabase.from_databricks(catalog=catalog, schema=schema, host=host_name, api_token=token, cluster_id=cluster_id)
    return DB_DATABASE
def get_analytics(id,flow,workspace):
    db = Database('workspace',"central")
    status,ppt_meta = db.fetch_one_record({"workspace_name":workspace})
    if status:
        if "host" in ppt_meta:
            # global_variable_declare(ppt_meta['catalog_name'],ppt_meta['schema_name'],ppt_meta['host'],ppt_meta['token'],ppt_meta['cluster_name'])
            spark = DatabricksSession.Builder().remote(host=ppt_meta['host'],token=ppt_meta['token'],cluster_id=ppt_meta['cluster_name']).getOrCreate()
            
            ddf=spark.sql(f"""SELECT CEIL(100 * 100 / COUNT(*)) AS sample_percent FROM {id};""")
            pandas_df = ddf.toPandas()
            print(pandas_df['sample_percent'][0])
            df = spark.sql(f"""select * from {id} TABLESAMPLE ({pandas_df['sample_percent'][0] if pandas_df['sample_percent'][0]<100 else 100} PERCENT) LIMIT 100""")
            pandas_df = df.toPandas()
            csv_string = pandas_df.to_csv(index=False)
            # print(csv_string)
            # table_json={"table_name":table_data.full_name,"columns":table_data.columns,"comments":table_data.comment,"constraints":table_data.table_constraints,"datasource_format":table_data.data_source_format}
            summary = summary_generation(csv_string)
            print(summary)
            return {
                "summary":summary,
                "keywords":[],
                "sample_questions":[]
            }
    return {
            "summary":"",
            "keywords":[],
            "sample_questions":[]
        }

@retry(Exception, tries=10, delay=1)
def summary_generation(input_metadata):
    try:
        value = random.randint(0, 1)
        AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
        openai.azure_endpoint = openai_api_base[value]
        openai.api_key = openai_key[value]
        messages = [{"role": "user", "content": f"Please provide a generalized summary of what the following table data talks about in atmost one paragraph with consistent summary and concise information based on fraction of table data, provide generalized information oly:\n\ntable data:{input_metadata}\n\nSummary:\n-"}]
        response = openai.chat.completions.create(
            model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=messages,
            temperature=0,
            max_tokens=250,
            timeout=60
        )
        text = response.choices[0].message.content
        if text:
            return text.strip()
        return ""   
    except Exception as error:
        print(f"Failed->summary_generation. Error {error}")

@retry(Exception, tries=10, delay=1)
def chat_title(chat_conversations):
    value = random.randint(0, 1)
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
    openai.azure_endpoint = openai_api_base[value]
    openai.api_key = openai_key[value]
    message = [{"role" : USER, "content" : f"Conversation start:\n{str(chat_conversations[-2:])}]=\nConversation end.\nSummarize the conversation in 5 words or fewer:"}]
    chat_completion = openai.chat.completions.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=message, 
        temperature=0, 
        max_tokens=35, 
        timeout=10)
    chat_content = chat_completion.choices[0].message.content.strip()
    return chat_content

def parse_citations(chat_history,feedbacks):
    chat_output = []
    for chat,feedback in zip(chat_history,feedbacks):
        chats = {"role":chat['role']}
        chats['data'] = []
        chat_meta={}
        if 'content' in chat:
            chat_meta={'content':chat['content']}
        if 'sql_query' in chat:
            chat_meta['sql_query']=chat['sql_query']
            chat_meta['data_location']=chat['data_location']
        if 'graph_json_url' in chat:
            # chat_meta['image_url']=chat['image_url']
            chat_meta['graphjson']=download_data(chat['graph_json_url'],'json')
        chats['data']=chat_meta
        

        chats['citations'] = []
        chats['feedback'] = feedback
        chat_output.append(chats)
    return chat_output

def retrieve_chats(chat_id,workspace):
    try:
        db = Database('AnalyticsCollection',workspace)
        status,record = db.fetch_one_record({'_id':chat_id})
        # print(record)
        chat_meta = record
        chat_conversations = chat_meta['chat_conversations']
        feedbacks = chat_meta['feedback']
        chat_meta['chat_conversations']=parse_citations(chat_conversations,feedbacks)[1:]
        feedbacks = chat_meta['feedback']
        for chat, analytics in zip(chat_meta['chat_conversations'], chat_meta['analytics'][1:]):
            chat.update(analytics)
        chat_meta.pop('analytics', None)
        chat_meta.pop('feedback', None)
        chat_meta['chat_id'] = chat_meta.pop('_id')
        return True,chat_meta
    except Exception as error:
        print(f'Failed chat_utils->retrieve_chat. Error:{error}')
        return False,"Failed to retrieve chat."
    
def toggle_feedback(chat_id,value,index,workspace):
    db = Database('AnalyticsCollection',workspace)
    status,record = db.fetch_one_record({'_id':chat_id},{'_id':0,'feedback':1})
    if status:
        feedback = record['feedback']
        feedback[index+1] = value
        status, message = db.update_one_record(chat_id,{"feedback":feedback})
        if status:
            return True,"Feedback Update Success."
        return True, message
    return False,"Feedback Update Failed."

def delete_chat(chat_id,workspace):
    db = Database('AnalyticsCollection',workspace)
    status,record = db.fetch_one_record(chat_id)
    status,message = db.delete_single_record(chat_id)
    if status:
        return True,"Successfully deleted chat record."
    return False,message

def category(date):
    current_date = datetime.now().date()
    diff = (current_date - date.date()).days
    if diff == 0:
        return 'Today'
    elif diff == 1:
        return 'Yesterday'
    elif diff < 8:
        return "Previous 7 Days"
    elif diff < 32:
        return "Previous 30 Days"
    return date.strftime('%B')

def fetch_chat_history(user_id,workspace):
    db = Database('AnalyticsCollection',workspace)
    status,record = db.fetch_aggregate([{"$match":{'user_id':user_id}},{"$project":{'title':{'$ifNull': [ "$title", None ] },'updated_on':{'$ifNull': [ "$updated_on", None ] },'chat_flow':{'$ifNull': [ "$chat_flow", None ] }}}])
    data = []
    if status and len(record)>0:
        df = pd.DataFrame(record)
        date_column = 'updated_on'
        df[date_column] = pd.to_datetime(df[date_column],format='%d-%m-%Y %H:%M:%S',errors='coerce')
        df.dropna(subset=[date_column], inplace=True)
        df['category'] = df[date_column].apply(category)
        df = df.sort_values(date_column,ascending=[False])
        df[date_column] = df[date_column].dt.strftime('%d-%m-%Y %H:%M:%S')
        categories = df.category.unique()
        if "Today" not in categories or len(categories)==0:
            data.append({'category':"Today",'records':[]})
        for cat in df.category.unique():
            cats = {}
            cats['category'] = cat
            cats['records'] = df[df['category']==cat].drop(['category'],axis=1).reset_index(drop=True).to_dict('records')
            data.append(cats)
    if len(data)==0:
        data.append({'category':"Today",'records':[]})
    return True, {"data":data}

def retrieve_chat(chat_id,workspace):
    try:
        db = Database('AnalyticsCollection',workspace)
        status,record = db.fetch_one_record({'_id':chat_id})
        chat_meta = record
        chat_conversations = chat_meta['chat_conversations']
        # citation = chat_meta['citations']
        feedbacks = chat_meta['feedback']
        chat_meta['chat_conversations'] = parse_citations(chat_conversations[-1:],feedbacks[-1:])
        for chat, analytics in zip(chat_meta['chat_conversations'], chat_meta['analytics'][-1:]):
            chat.update(analytics)
        chat_meta.pop('analytics', None)
        chat_meta.pop('feedback', None)
        chat_meta['chat_id'] = chat_meta.pop('_id')
        return True,chat_meta
    except Exception as error:
        print(f'Failed chat_utils->retrieve_chat. Error:{error}')
        return False,"Failed to retrieve chat."

def get_schema_information(host,token,catalog_name, schema_name, table_name):
    table_full_name = catalog_name+"."+schema_name+"."+table_name
    table_name=[]
    api_client = WorkspaceClient(host=host, token=token)


    constraints = api_client.tables.get(table_full_name).as_dict()
    if "table_constraints" in constraints:
        table_constraints = constraints["table_constraints"]
    else:
        table_constraints={}
    columns=list(map(lambda schema: {"column_name":schema['name'],"data_type":schema['type_name']},constraints["columns"]))
    final={"columns":columns, "table_constraints": table_constraints}
    return final

def generate_stream_response(query,user_id,chat_id,workspace):
    try:
        
        db=Database("workspace","central")
        db = Database('workspace',"central")
        status,data_workspace = db.fetch_one_record({"workspace_name":workspace})
        host_name=data_workspace["host"]
        token=data_workspace["token"]
        catalog_name=data_workspace["catalog_name"]
        schema_name=data_workspace["schema_name"]
        cluster_id=data_workspace['cluster_name']
        # print(host_name,token,catalog_name,schema_name,cluster_id)
        
        now = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        today_date=datetime.now().strftime('%Y-%m-%d')

        db = Database('AnalyticsCollection',workspace)
        
        chat_meta = {}
        filter = None
        
        status,record = db.fetch_one_record({'_id':chat_id})
        chat_meta.update(record)
        chat_conversations = chat_meta['chat_conversations']
        citation = chat_meta['citations']
        analytics = chat_meta['analytics']
        feedback = chat_meta['feedback']
        table=chat_meta['chat_table']
        table_schemas_json = get_schema_information(host=host_name,token=token,catalog_name=catalog_name, schema_name=schema_name,table_name=table)
        if 'dates' in chat_meta:
            dates = chat_meta['dates']
        print('Chat history retrieved.')

        if len(query.strip())>0:

            feedback.append(None)
            citation.append([])
            analytics.append({
                    "summary":"",
                    "keywords":[],
                    "sample_questions":[]
                    })
            if len(dates):
                dates.append({"query_date": today_date,"feedback_date": None})



            input_text = f"User Query:\n{query}\n"

            messages = [{"role": "system", "content": system_message_conversation.format(table=table,table_data=table_schemas_json)}]
            messages.append({"role": "user", "content": input_text})
        
            chat_content = ''
            print(cluster_id)
            start=time.time()
            
            DB_DATABASE=global_variable_declare(catalog_name,schema_name,host_name,token,cluster_id)
            end=time.time()
            print("Databricks connection time",end-start)
            for status,response in stream_table_response(messages,DB_DATABASE):
                chat_content+=response
                if status:
                    time.sleep(0.1)
                    yield 'stream',json.dumps({"text":response})
            print("Repo response streaming completed")
            print(chat_content)
            
            print("Extracted citations from repo index reponse.")
            print(chat_content)
            
        else:
            chat_conversations.append({"role":USER,"content" : query})
            chat_content = "I apologize, but it seems like your message is empty or unclear. Please provide a valid question or input, and I'll be happy to assist you. If you need any help or have any inquiries, feel free to ask."
        chat_conversations.append({"role": USER, "content": query})
        chat_conversations.append({"role":ASSISTANT, "content": chat_content})
       
        citation.append([])
        analytics.append({
                "summary":"",
                "keywords":[],
                "sample_questions":[]
                })
        feedback.append(2)
        if len(dates):
            dates.append({"query_date": None,"feedback_date": None})

        print("Updated user query response.")


        chat_meta['chat_conversations'] = chat_conversations
        chat_meta['citations'] = citation
        chat_meta['analytics'] = analytics
        chat_meta['updated_on'] = now
        chat_meta['feedback'] = feedback
        chat_meta['dates'] = dates

        status,message = db.update_one_record(chat_id,chat_meta)
        if not status:
            yield False, message
        print("Updated records.")
        chat_meta = {'_id':chat_meta['_id']}
        yield True,json_util.dumps(chat_meta)

    except Exception as error:
        print(f'Failed chat_utils->generate_response. Error:{error}')
        yield False,"Failed to retrieve chat/response"

def stream_table_response(chat_conversations,DB_DATABASE):
    value = random.randint(0, 1)
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
    openai_client = AzureChatOpenAI(
        azure_endpoint = openai_api_base[value],
        api_version="2023-09-01-preview",
        api_key = openai_key[value],
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        temperature=0,
        max_tokens=1024*4,
        max_retries=10,
        streaming=True
    )

    FORMAT_INSTRUCTIONS = """Use the following format:
        Question: the input question you must answer, get user question before first ','
        Thought: you should always think about what to do based on table relationship
        Action: the action to take, only and use one of tools[sql_db_schema,], apply limit 20 in every query
        use this both information
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N time)
        Thought: I now know the final answer
        Final Answer: Give final answer in english language only
        If you can not find user query related answer then return, this table does not contain information regarding your query"""
    
    toolkit=SQLDatabaseToolkit(db=DB_DATABASE, llm=openai_client)
    start=time.time()
    agent = create_sql_agent(llm=openai_client,toolkit=toolkit,top_k=10,verbose=True,agent_executor_kwargs={"handle_parsing_errors": True})
    data=agent.run({"input":chat_conversations})
    end=time.time()
    print("Agent running time",end-start)
    data=data.split()
    for token in data:
        yield True,token+" "

def download_data(file_location,file_type):
    storage_uri = client.get_secret("STORAGE-ACCOUNT").value
    
    container_name = client.get_secret("CONTAINER-NAME").value
    
    blob_service_client = BlobServiceClient.from_connection_string(client.get_secret('STORAGE-CONNECTION-STRING').value)
    container_client=blob_service_client.get_container_client(container_name)
    blob_name=file_location.split(container_name+'/')[1]
    blob_client = container_client.get_blob_client(blob_name)
    blob_data = blob_client.download_blob().readall()
    if file_type=='csv':
        csv_data = blob_data.decode('utf-8')
        csv_buffer = StringIO(csv_data)
        df = pd.read_csv(csv_buffer)

        return df
    if file_type=='image':
        image = base64.b64encode(blob_data).decode("utf-8")

        return image
    if file_type=='json':
        json_data = blob_data.decode('utf-8')
        data = json.loads(json_data)

        return data

def upload_csv(user_id,df):

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    main_dir = f"SDA/{user_id}/csv_data"
    storage_uri = client.get_secret("STORAGE-ACCOUNT").value
    
    container_name = client.get_secret("CONTAINER-NAME").value
    
    blob_service_client = BlobServiceClient.from_connection_string(client.get_secret('STORAGE-CONNECTION-STRING').value)
    container_client=blob_service_client.get_container_client(container_name)

    blob_list = list(container_client.list_blobs(name_starts_with=main_dir))

    file_count = len(blob_list)+1
    file_name=f'user_query{file_count}.csv'
    file_path=main_dir+'/'+file_name
    blob_client = container_client.get_blob_client(file_path)

    blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True, encoding='utf-8')
    fileurl = storage_uri+container_name+'/'+file_path
    return fileurl

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

def natural_language_to_query(data,user_query,workspace_name,error_message):

    value = random.randint(0, 1)
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
    openai.azure_endpoint = openai_api_base[value]
    openai.api_key = openai_key[value]
    system_message_conversation=f"""{error_message}\nRole: You are a highly skilled data analyst specialized in generating SQL queries with precision and accuracy.

    Task: Your objective is to generate a detailed and correct Databricks SQL query that directly answers the analytical question provided. This query must be optimized for performance and correctness, following best practices.

    Analytical Question: {user_query}

    Schemas and Data: {data}

    Instructions (Follow These Carefully):

    1. Clarity and Accuracy:
    If User asks for sales consider it as net sales only.

    While selecting columns for the SQL query, Column description should have higher priority each timee, consider column description of each columns before picking up the column, and columns which satisfies the user query completely select only those columns.

    Ensure the SQL query is highly accurate and addresses the analytical question directly. Each part of the query should contribute to achieving the correct result.

    Re-read the analytical question before crafting the query to ensure full comprehension.

    2. Column and Table Names:

    Use column names exactly as they appear in the provided schema.
    If user asks Year on Year data then include case when like this, CASE WHEN MONTH(Date_column) >= 4 THEN YEAR(Date_column) ELSE YEAR(Date_column) - 1 END.
    If a user requests to calculate quarterly data, please note that the financial year starts in April, and each quarter consists of 3 months. quarter 1(Q1) is April to june, quarter 2(Q2) is july to september, quarter 3(Q3) is october to december and quarter 4(Q4) is january to march.
    Do not enclose table names in backticks; only column names should be enclosed in backticks.
    Avoid unnecessary complexity by only using the necessary tables and columns.

    3. Table Joins:

    Avoid unnecessary joins. If all required columns are in one table, do not apply joins.
    Apply joins only if required columns are located in different tables, ensuring the query is efficient.

    4.GROUP BY Clause:

    Thoroughly analyze whether a GROUP BY clause is required to answer the question. 

    5. Sorting:
    Ensure that the column used for sorting is included in the selected columns. If all the selected columns are of string data type, avoid applying sorting unless explicitly requested by the user.
    In the query check which columns have datatype double or integer. Also that column should be available  Only use those columns in sorting. ALWAYS sort data in descending order.

    6. Data Types and Formatting:

    For columns likely to be integers or doubles, cast them to double and round off values to two decimal places to ensure accuracy. If for some integer or double column data is num the use COALESCE().

    7. WHERE Clause:

    Apply a WHERE condition only if required.
    NEVER USER '=' IN WHERE CLAUSE when column datatype id string.
        
    To filter a string column based on a specified value, use the following SQL expression:
    INSTR(REGEXP_REPLACE(UPPER(column_name), '[^A-Z0-9]', ''), REGEXP_REPLACE(UPPER('User query specified value to filter data with'), '[^A-Z0-9]', '')) > 0
    
    If you need to filter the column using multiple values, apply the same function with the OR operator between each condition.
    DO NOT USE ANY OTHER FUNCTION TO COMPARE USER DEFINED VALUE OTHER THEN INSTR TO FILTER string column
    

    8. String Handling:

    In string comparisons, remove unnecessary characters like spaces, dashes, slashes, and asterisks using REGEXP_REPLACE() on both sides of the comparison.
    Convert both the column and value to uppercase using UPPER() in the WHERE clause to ensure consistency.

    9. Focus on Output:

    Your response should be the SQL query only. Do not include explanations, descriptions, or comments.

    Final Check:

    Review the query to ensure it is concise, correct, and optimized for performance before submission."""
    
    response = openai.chat.completions.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=[{'role': 'user', 'content': system_message_conversation}],
        temperature=0,
        max_tokens=1024,
        timeout=100
    )

    return response.choices[0].message.content,{"input":response.usage.prompt_tokens,"response":response.usage.completion_tokens}

def create_new_chat(workspace_name,user_id,user_query,sql_query,chat_id,location):
    chat_meta = {}
    db_analytics = Database('AnalyticsCollection',workspace_name)
    now = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    today_date=datetime.now().strftime('%Y-%m-%d')
    if chat_id == None: 
        chat_conversations = [{"role" : SYSTEM, "content" : system_message_chat_conversation},{"role":USER,"content" : user_query},{"role":ASSISTANT,"sql_query":sql_query,"data_location":location}]
        chat_meta['user_id'] = user_id
        chat_meta['created_on'] = now
        chat_meta['chat_flow'] = 'general'

        chat_meta['chat_conversations'] = chat_conversations
        chat_meta['citations'] = [[],[],[]]
        chat_meta['analytics'] = [{"summary":"","keywords":[],"sample_questions":[]},{"summary":"","keywords":[],"sample_questions":[]},{"summary":"","keywords":[],"sample_questions":[]}]
        chat_meta['updated_on'] = now
        chat_meta['feedback'] = [None,None,2]
        chat_meta['dates'] = [{"query_date": None,"feedback_date": None},{"query_date": today_date,"feedback_date": None},{"query_date": None,"feedback_date": None}]
        # chat_meta['sql_query']=['',sql_query,'']
        chat_meta["title"] = chat_title(chat_conversations[-1:])
        status,id = db_analytics.insert_single_record(chat_meta)
        if not status:
            return False,""
    else:
        status,record = db_analytics.fetch_one_record({'_id':chat_id})
        chat_meta.update(record)
        chat_conversations = chat_meta['chat_conversations']
        citation = chat_meta['citations']
        analytics = chat_meta['analytics']
        feedback = chat_meta['feedback']
        # sql_query=chat_meta['sql_query']
        if 'dates' in chat_meta:
            dates = chat_meta['dates']
        chat_conversations=chat_conversations+[{"role":USER,"content" : user_query},{"role":ASSISTANT,"sql_query":sql_query,"data_location":location}]
        citation=citation+[[],[]]
        analytics=analytics+[{"summary":"","keywords":[],"sample_questions":[]},{"summary":"","keywords":[],"sample_questions":[]}]
        feedback=feedback+[None,2]
        if len(dates):
            dates=dates+[{"query_date": today_date,"feedback_date": None},{"query_date": None,"feedback_date": None}]
        # sql_query.append([sql_query,''])
        chat_meta['chat_conversations'] = chat_conversations
        chat_meta['citations'] = citation
        chat_meta['analytics'] = analytics
        chat_meta['updated_on'] = now
        chat_meta['feedback'] = feedback
        chat_meta['dates'] = dates
        # chat_meta['sql_query']=sql_query

        status,message = db_analytics.update_one_record(chat_id,chat_meta)
        if not status:
            False, message
        id=chat_id
    return True,id,chat_meta['updated_on'],chat_meta["title"],chat_meta['chat_flow']

@retry(Exception, tries=10, delay=1)
def query_generation(query,workspace_name,chat_id):

    value = random.randint(0, 1)
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
    openai.azure_endpoint = openai_api_base[value]
    openai.api_key = openai_key[value]
    
    db_analytics = Database('AnalyticsCollection',workspace_name)
    status,record = db_analytics.fetch_one_record({'_id':chat_id})

    
    chat_conversations=record['chat_conversations']
    if len(chat_conversations)>4:
        mod_chat_history = chat_conversations[-4:]
    else:
        mod_chat_history = chat_conversations[1:]

    keys_to_remove = {"data_location","graph_json_url","sql_query"}

    # Remove specified keys using map and dictionary comprehension
    mod_chat_history = list(
        map(lambda item: {k: v for k, v in item.items() if k not in keys_to_remove}, mod_chat_history)
    )
    
       
    task = """
        Identify if the Current User Query is independent or dependent.
        If you think the Current User Query is independent and does not require any completion simply return the Current User Query as the Modified Query.
        If you think the Current user query is dependent on previous conversation then only and only then try to remove this dependency from the Current User Query by taking relevant context from the Chat History and return the independent query as the Modified Query.
        Donot remove mentions of any specific values table names in the original query. Keep them in the modified query as well.
    """
    msg = [{"role": "user", "content": f"Chat History:{mod_chat_history}\nCurrent User Query: {query}\nSystem Task:{task}\nModified Query (if necessary):"}]
    

    completion = openai.chat.completions.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=msg,
        temperature=0,
        max_tokens=200,
        timeout=20)
    result = completion.choices[0].message.content
    print(result)
    return result,{"input":completion.usage.prompt_tokens,"response":completion.usage.completion_tokens}

def identify_visualization(df,user_query):

    value = random.randint(0, 1)
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
    openai.azure_endpoint = openai_api_base[value]
    openai.api_key = openai_key[value]

    system_message_conversation = f"""You are a highly skilled data visualization assistant. Your task is to generate the appropriate type of graph based on the user's query and the provided CSV data. If generating a graph is not possible, respond with "sorry" without any explanation.
    1. Analyze the user's query to determine the type of graph they are requesting.
    2. Use the provided CSV data to predict the most possible graph type.
    3. Return only the graph Type as a response.
    4. Return graph type which can be possible to create using plotly. 
    4. If there is only one row/column in the csv data then return "Sorry."
    5. If a graph cannot be generated, return "Sorry."

    User Query: {user_query}
    CSV Data: 
    {df}
    """

    response = openai.chat.completions.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=[{'role': 'user', 'content': system_message_conversation}],
        temperature=0,
        max_tokens=1024,
        timeout=60
    )
    content=response.choices[0].message.content
    print(content)
    return content,{"input":response.usage.prompt_tokens,"response":response.usage.completion_tokens}

def check_graph_possibility(user_query,workspace_name,user_id,chat_id):
    logger_params={}
    start = datetime.now()
    # logger_params['funcName'] = "check_graph_possibility"
    logger_params['query'] = user_query
    logger_params['user_id'] = user_id

    logger_params['funct_start_time'] = start
    logger_params['input'] = 0
    logger_params['response'] = 0
    db_central=Database("workspace","central")
    print(workspace_name)
    status_central,info=db_central.fetch_one_record({"workspace_name":workspace_name},{"schema_name":1,"catalog_name":1,"cluster_name":1,"host":1,"token":1})
    # print(info)
    db=Database('tebleMetadata',workspace_name)
    status_db,data=db.fetch_all_records({"schema_name":info['schema_name'],"catalog_name":info['catalog_name']},{"_id":0})
    query_token={}
    mod_user_query,query_token =query_generation(user_query,workspace_name,chat_id) if chat_id!=None else (user_query,{'input':0,'response':0})
    if query_token:
        logger_params['input'] += int(query_token['input'])
        logger_params['response'] += int(query_token['response'])
    max_retries = 3
    attempts = 0
    error_message=''
    while attempts < max_retries:
        generated_queries=[]
        txt2sql = {}
        query,txt2sql=natural_language_to_query(data,mod_user_query,workspace_name,error_message)
        
        generated_queries.append(query)
        print(query)
        try:
            df=query_to_dataframe(info['host'],info['token'],query.replace('```sql','').replace('```',''),info['catalog_name'],info['schema_name'],info['cluster_name'])
            df = df.dropna(how='all')
            if df.empty:
                error_message=f"""You are a highly skilled data analyst. Your task is to analyze the user query again and modify the WHERE conditions to ensure the resulting SQL query returns data. Follow these instructions:
                1. Carefully review the user's query to understand the intent and the conditions applied. Reiterate selected tables columns, If column name is not clearly mentioned in user query.
                2. Modify the WHERE conditions in the SQL query to prevent an empty dataframe, ensuring the query is different from any previously generated queries.
                3. Do not generate a query that is exactly same as any of the previously generated queries. DO NOT REPEAT PREVIOUSLY GENERATED QUERIES IN NEWLY REGENERATED QUERY.
                Previously Generated Queries: {generated_queries}\n\n"""
                attempts+=1
                continue
            break
        except Exception as e:
            print(e)
            error_message=f"""Getting error while running above generated query: {query}\n\n error: {repr(e)}\n\nRegenerate query by keep in mind below instructions.\n3. Do not generate a query that is exactly same as any of the previously generated queries.\nPreviously Generated Queries: {generated_queries}\n\n"""
            attempts += 1
            df=pd.DataFrame([],columns=["no data available"])
    if txt2sql:
        logger_params['input'] += int(txt2sql['input'])
        logger_params['response'] += int(txt2sql['response'])
    df=df.drop_duplicates(keep='first')
    file_path=upload_csv(user_id,df)
    status_new_chat,result_new_chat,updated_on,title,chat_flow=create_new_chat(workspace_name,user_id,user_query,query.replace('```sql','').replace('```',''),chat_id,file_path)
    visual = {}
    content,visual=identify_visualization(df,mod_user_query)
    if "Sorry" in content:
        graph_type=2
    elif "Sorry" not in content:
        graph_type=3
    if visual:
        logger_params['input'] += int(visual['input'])
        logger_params['response'] += int(visual['response'])
    end = datetime.now()
    logger_params['time_taken'] = (end - start).total_seconds()
    LOGGER.logger.debug("Decision making finished...", extra=logger_params)
    return True,{"user_query":mod_user_query,"chat_id":result_new_chat,"type":graph_type,"sql_query":query.replace('```sql','').replace('```',''),"csv_data":file_path,"graph_type":content if "Sorry" not in content else None,"updated_on":updated_on,"title":title,"chat_flow":chat_flow}

def save_data(user_id,data,file_type):
    storage_uri = client.get_secret("STORAGE-ACCOUNT").value
        
    container_name = client.get_secret("CONTAINER-NAME").value
    
    blob_service_client = BlobServiceClient.from_connection_string(client.get_secret('STORAGE-CONNECTION-STRING').value)
    container_client=blob_service_client.get_container_client(container_name)

    if file_type=='image':
        
        main_dir = f"SDA/{user_id}/plot_image"
        blob_list = list(container_client.list_blobs(name_starts_with=main_dir))
        file_count = len(blob_list)+1
        file_name=f'graph{file_count}.png'
        file_path=main_dir+'/'+file_name
    if file_type =='json':
        main_dir = f"SDA/{user_id}/plot_json"
        blob_list = list(container_client.list_blobs(name_starts_with=main_dir))
        file_count = len(blob_list)+1
        file_name=f'graph{file_count}.json'
        file_path=main_dir+'/'+file_name
    blob_client = container_client.get_blob_client(file_path)

    blob_client.upload_blob(data, overwrite=True)
    fileurl = storage_uri+container_name+'/'+file_path
    return fileurl


def create_visualisation(user_id,workspace_name,chat_id,file_location,graph_type):
    if graph_type:
        logger_params={}
        start = datetime.now()
        logger_params['user_id'] = user_id

        logger_params['funct_start_time'] = start
        logger_params['input'] = 0
        logger_params['response'] = 0
        df=download_data(file_location,'csv')
        df=df.dropna()
        df=df.head(20)
        pipe_separated_string = df.to_csv(sep='|', index=False)

        value = random.randint(0, 1)
        AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
        openai.azure_endpoint = openai_api_base[value]
        openai.api_key = openai_key[value]

        system_message_conversation=f""" You are a highly skilled data visualization assistant. Your task is to generate Python code for creating a {graph_type} using the Plotly library based on the provided dataframe {df}. Follow these instructions:
        1. Create a {graph_type} using the Plotly library.
        2. Generate Python code for creating the visualization, using data from the {pipe_separated_string} dataframe.
        3. All key arrays must be of same length in graph data json.
        3. Try to consider all columns in the graph. and all columns data should be coming from the above given data.
        4. First Sort X-axis values into ascending order then Cast x-axis data into string.
        5. Only and only provide Python code without instructions and do not add Python in the start and in Python code.
        6. Do not show the figure. The figure variable should always be 'fig'.
        7. DO NOT ADD ANY EXTRA INFORMATION OTHER THEN PYTHON CODE
            """
        response = openai.chat.completions.create(
            model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=[{'role': 'user', 'content': system_message_conversation}],
            max_tokens=1024*3,
            temperature=0,
            timeout=120
        )
        content=response.choices[0].message.content
        print(content)
        logger_params['input'] = response.usage.prompt_tokens
        logger_params['response'] = response.usage.completion_tokens
        content1 = content.replace("```python", "").replace("```", "")
        exec(content1,globals())

        graphJSON = json.dumps(globals()['fig'], cls=plotly.utils.PlotlyJSONEncoder)
        json_url=save_data(user_id,graphJSON,'json')
        graphJSON=json.loads(graphJSON)
        

        chat_meta={}
        db_analytics = Database('AnalyticsCollection',workspace_name)
        analytics_status,analytics_record = db_analytics.fetch_one_record({'_id':chat_id})
        chat_meta.update(analytics_record)
        chat_meta['chat_conversations'][-1]['graph_json_url']=json_url
        status_update,message = db_analytics.update_one_record(chat_id,chat_meta)
        end = datetime.now()
        logger_params['time_taken'] = (end - start).total_seconds()
        LOGGER.logger.debug("visualization finished...", extra=logger_params)

        return True,graphJSON
    else:
        return False,"Graph Type is not mentioned"

def num_tokens_from_messages(messages):
    encoding= tiktoken.get_encoding("cl100k_base")  
    num_tokens = 0
    for message in messages:
        num_tokens += 4  
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  
                num_tokens += -1  
    num_tokens += 3  
    return num_tokens

def analyse_data(user_id,workspace_name,chat_id,file_location,graph_type,user_query,sql_query):
    logger_params={}
    start = datetime.now()
    logger_params['user_id'] = user_id

    logger_params['funct_start_time'] = start
    logger_params['input'] = 0
    logger_params['response'] = 0
    value = random.randint(0, 1)
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
    openai.azure_endpoint = openai_api_base[value]
    openai.api_key = openai_key[value]

    db_analytics = Database('AnalyticsCollection',workspace_name)
    analytics_status,analytics_record = db_analytics.fetch_one_record({'_id':chat_id})

    df=download_data(file_location,'csv')
    df=df.drop_duplicates(keep="first")
    df=df.head(20)
    pipe_separated_string = df.to_csv(sep='|', index=False)
    json_string = df.to_json(orient='records')
    dict_result = df.to_dict(orient='list')
    print(dict_result)

    system_message_conversation=f"""Role: You are an expert data analyst.
    Objective: Provide a comprehensive response to the user’s query based on the provided data.
    Follow all below 10 Instructions:

    1. Interpret the user's question carefully to identify the specific information they seek.
    2. In the response wather it is a tabular response or textual response, All rows and column data should be there.
    3. If the provided data does not contain relevant answer as per the user question, then provide answer with given data and some other information.
    3. Generate a clear and descriptive response from the data that answers the user's query. Do not filter out any rows or columns, as all the data provided is accurate and should be included in the response. 
    4. If the data related to sales is provided, please note that it is denominated in rupees.
    5. If the data is not empty and the user has not asked for a table format, provide a textual response using insights from the data.
    6. If the user requests a table format then only, provide the data in a pipe-separated table format enclosed between <table> and </table> tags. and instead of giving an empty row "|------|-------|" after the header enclose table headers between <thead>| header1 | header2 |</thead> (determine the headers) and table body in <tbody>| cell1 | cell2 |\n| cell3 | cell4 |</tbody> donot use any other tags (like <tr> or <td>) aside from these.
    7. Always provide precise textual insights based on the data each time.
    8. If the user specifies a graph type, focus solely on the analysis and insights without mentioning visualization.
    9. If the data is empty or all values are null, then and then only respond with "Data not available for this query."
    10. Do not include unnecessary explanations in the response.
    
    Data: {json_string}
    Given Data is being derived from given sql Query: {sql_query}
    query: {user_query}"""
    messages=[{'role': 'user', 'content': system_message_conversation}]
    
    response = openai.chat.completions.create(
            model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=messages,
            temperature=0,
            max_tokens=1024,
            timeout=100,
            stream=True
        )
    logger_params['input'] = num_tokens_from_messages(messages)
    content=''
    for chunk in response:
        if chunk is not None and len(chunk.choices):
            # print(chunk)
            content+=chunk.choices[0].delta.content if chunk.choices[0].delta.content!=None else ""
            yield 'stream',json.dumps({"text":chunk.choices[0].delta.content}) if chunk.choices[0].delta.content!=None else json.dumps({"text":""})
    # content=response.choices[0].message.content
    chat_meta={}
    chat_meta.update(analytics_record)
    logger_params['response'] = num_tokens_from_messages([{"role":"assistant","content":content}]) 
    chat_content = convert_pipe_to_html(content)
    chat_meta['chat_conversations'][-1]['content']=chat_content
    end = datetime.now()
    logger_params['time_taken'] = (end - start).total_seconds()
    LOGGER.logger.debug("Analysis finished...", extra=logger_params)
    status_update,message = db_analytics.update_one_record(chat_id,chat_meta)

def convert_pipe_to_html(text):
    # Define regex patterns for <thead> and <tbody>
    thead_pattern = r'<thead>(.*?)</thead>'
    tbody_pattern = r'<tbody>(.*?)</tbody>'
    table_pattern = r'<table>(.*?)</table>'
    
    # Convert thead content
    def convert_thead(match):
        header_content = match.group(1)
        rows = header_content.strip().split('\n')
        headers = rows[0].strip().split('|')[1:-1]  # Extracting headers, ignoring first and last empty strings
        header_html = '<tr>' + ''.join(f'<th scope="col">{header.strip()}</th>' for header in headers) + '</tr>'
        return f'<thead>{header_html}</thead>'
    
    # Convert tbody content
    def convert_tbody(match):
        body_content = match.group(1)
        rows = body_content.strip().split('\n')
        body_html = ''
        for row in rows:
            cells = row.strip().split('|')[1:-1]  # Extracting cells, ignoring first and last empty strings
            body_html += '<tr>' + ''.join(f'<td scope="row">{cell.strip()}</td>' for cell in cells) + '</tr>'
        return f'<tbody>{body_html}</tbody>'
    
    # Convert tables
    def convert_table(match):
        table_content = match.group(1)
        table_content = re.sub(thead_pattern, convert_thead, table_content, flags=re.DOTALL)
        table_content = re.sub(tbody_pattern, convert_tbody, table_content, flags=re.DOTALL)
        return f'<div class="table-responsive"><table class="table table-hover table-striped">{table_content}</table></div>'
    
    # Perform conversions
    text = re.sub(table_pattern, convert_table, text, flags=re.DOTALL)
    
    return text
