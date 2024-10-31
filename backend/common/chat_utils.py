import os
import re
import time
import copy
import json
import openai
import random
import logging
import tiktoken
import requests
import pandas as pd
from retry import retry
from datetime import datetime
from bson import json_util, ObjectId
from common.database import Database
from common.utils import (ppt2img_thumb, 
                          pdf2img_thumb, 
                          delete_blob, 
                          remove_from_index,
                          generate_link,
                          run_job
                        )
from common.keyvault_connection import get_conn


client=get_conn()

GPT_TYPE = client.get_secret("GPT-TYPE").value

MASTER_TABLE = "chat_document_collection"
MASTER_TABLE_COLUMNS = ['primary_id', 'oid', 'content', 'chunk_number', 'file_name', 'display_name', 'file_path', 'image_path','modified_on']
REPO_TABLE = "chat_document_summary_collection"
REPO_TABLE_COLUMNS = ['primary_id', 'oid', 'file_name', 'content', 'file_type','file_path','modified_on']
CATALOG = "databricks_gptexponent_catalogue"
DATABRICKS_URL = client.get_secret("DATABRICKS-URL").value
LOCAL_CHAT_TABLE = "local_chat_document_collection"
LOCAL_CHAT_TABLE_COLUMNS = ['primary_id', 'oid', 'content', 'chunk_number', 'file_name', 'display_name', 'file_path', 'image_path','modified_on']

CACHE_TABLE = "chat_cache"
CACHE_TABLE_COLUMNS = ["id","oid","query","created_on","user_id"]

## PAT_TOKEN needs to be updated after it expires.
PAT_TOKEN = client.get_secret("DATABRICKS-PAT").value

openai.api_type = "azure"
openai.api_version = "2023-05-15"

#Azure search indexes
kb_index = "{}-gptxponent-chat"
repo_index = "{}-gptxponent-repo"
local_index = "{}-gptxponent-local"


if GPT_TYPE == "GPT-4":
    openai_api_base = [client.get_secret("AZURE-OPENAI-ENDPOINT-US").value,client.get_secret("AZURE-OPENAI-ENDPOINT-US").value]
    openai_key = [client.get_secret('OPENAI-API-KEY-US').value,client.get_secret('OPENAI-API-KEY-US').value]
    DEPLOYMENT_NAME = [client.get_secret("CHAT-MODEL").value, client.get_secret("CHAT-MODEL").value]
else:
    openai_api_base = [client.get_secret("AZURE-OPENAI-ENDPOINT-US").value,client.get_secret("AZURE-OPENAI-ENDPOINT-FRANCE").value]
    openai_key = [client.get_secret('OPENAI-API-KEY-US').value,client.get_secret('OPENAI-API-KEY-FRANCE').value]
    DEPLOYMENT_NAME = ["chatgpt-16k", "chatgpt-16k"]

SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"

max_response_tokens = 1024
token_limit = 1024*20
user_msg = "Do not justify your answers. Do not give information not mentioned in the CONTEXT INFORMATION."
assistant_msg = "Sure! I will stick to all the information given in the system context. I won't answer any question that is outside the context of information. I won't even attempt to give answers that are outside of context. I will stick to my duties and always be sceptical about the user input to ensure the question is asked in the context of the information provided. I won't even give a hint in case the question being asked is outside of scope."

system_message_chat_conversation = """ 
Welcome to GPTXponent, where we specialize in document analysis and information extraction. Please provide your query, and I will assist you by referencing our extensive knowledge base.
 
When formulating your response, adhere to the following guidelines:

Keep you answers as concise as possible with no additional information unrelated to the questions asked. Include all citations in the response you give.
 
If response contains any tables give it in a pipe separated table format and enclose the table between <table> </table> tags and instead of giving an empty row "|------|-------|" after the header enclose table headers between <thead>| header1 | header2 |</thead> (determine the headers) and table body in <tbody>| cell1 | cell2 |\n| cell3 | cell4 |</tbody> donot use any other tags (like <tr> or <td>) aside from these.

When preparing your response, please include tables (using above formating) and apply appropriate styling where necessary to enhance clarity and presentation.

If a user requests to calculate quarterly data, please note that the financial year starts in April, and each quarter consists of 4 months.

If the information is uncertain or not available within our sources, clearly state "I don't have a response" or "I don't know the answer"

If the user query is regarding the competitors use the csv tables present in the sources to get the correct competitor names (Donot include Knockout before competitor names)

Include every source that is used to answer any query in the citations. Donot make up answers on your own and strictly answer using the sources provided only.

1. Source-Based Responses: Ensure each piece of information in your response is directly linked to a specific piece of source content. Use the quadruple angle brackets format "<<<<Source Name>>>>" at the end of each relevant sentence or paragraph to cite the source. 
                           Following below scenario for source citation format:
                            Structure of Input Source Content:
                            Tds Feviseal Neutral Pro Clear (I)_Page_1: 2023-11-23T11:32:29Z (modified_on date): context
                            So, Source-Based Response should be <<<<Tds Feviseal Neutral Pro Clear (I)_Page_1>>>> and do not include modified_on date in response.
 
2. Distinct Source Citations: Avoid merging information from different sources in a single statement. Provide separate citations for each source, maintaining clear distinctions between them. Use "<<<<Source Name>>>>" for each citation.
 
3. Source Recency and Relevance: When multiple sources contain overlapping information, prioritize sources based on a combination of recency and relevance to the query. While recent sources are often preferred to ensure up-to-date information, also consider the depth, specificity, and context provided by older sources. Use your judgment to decide which source(s) offer the most comprehensive and pertinent information for the user's query, and cite accordingly using "<<<<Source Name>>>>". In cases where older sources provide valuable insights or historical context that enrich the response, include these citations alongside the most recent sources.
 
4. Formal Tone: Maintain a professional and formal tone throughout your response. Structure your response appropriately, using paragraphs, bullet points, or tables as the information dictates.
 
5. Detailed Reasoning: For responses involving calculations or estimations, include a detailed explanation of your methodology and reasoning, ensuring transparency in how conclusions are reached. Cite your sources with "<<<<Source Name>>>>".

    If the question is related to calculations, understand and provide detailed analysis. Provide step by step calculations.
 
6. Contextual Responses: For responses derived from the context of the conversation or implied knowledge, ensure that each piece of information is accompanied by a citation to a specific source where the information can be verified. Use "<<<<Source Name>>>>" for these citations as well.
 
7. Response Clarity: If the information is uncertain or not available within our sources, clearly state "I don't have a response" or "I don't know the answer," to maintain the integrity of our information.
 
8. Feedback Loop: After providing a response, encourage the user to provide feedback or ask follow-up questions. This will help in refining the system's accuracy and effectiveness over time.
 
9. Query Guidance: If a user's query is unclear or could be refined for better results, gently guide them on how to rephrase their query for more effective results.
 
Your adherence to these guidelines ensures that our responses are accurate, reliable, and valuable to our users.
"""

system_message_repo_conversation="""
You are a powerful knowledge base assistant. You can answer questions about the knowledge base. Must follow all the instructions below on how to behave.
Instruction 1: Compose an answer using knowledge base sources given along with the user question.
Instruction 2: Answer exactly with the given source information based on the user question.
Instruction 3: Each source is has the format of file_name: summary where summary contains small summary of data, always include the source name for each fact you use in the response. Use quadruple angle brackets to reference the source, e.g. <<<<file_name>>>> User file_name for each facts from it respective sources.
Instruction 4: Please make sure that the response should always give source reference at the end of each sentence. Please don't ignore this instruction.
Instruction 5: Your answer response should always have source reference for each sentence in this format <<<<file_name>>>>. Include timestamp with the filename in response where ever mentioned.
Instruction 6: Don't combine sources, list each source separately, e.g. <<<<file_name 1>>>> <<<<file_name 2>>>>.
Instruction 7: Your response source references should come from knowledge base source names and keep validated source name as reference in quadruple angle brackets. Please don't ignore this instruction.
Instruction 8: Pick latest information to answer the questions by choosing given sources with modified_on details 
Instruction 9: Your behavior should be formal and professional.
Instruction 10: If your source references point to only one source, please make sure to give source reference at the end of the response.
Instruction 11: The response should include detailed information and be well formatted into paragraphs, points, tables, sections, etc. according to the user question.
Instruction 12: Links should not be included anywhere other than the citation.
Instruction 13: If you think your response may be inaccurate or vague, do not write it and answer with the exact text "I don't have a response."
Instruction 14: If you don't know the answer OR can't find the source information, you should answer with the exact text "I don't know the answer."
Instruction 15: Prioritize information from the most recent source by considering the modified_on value when formulating your response.
Instruction 16: If the user's query is related to showing a video, image, or audio about a specific product or topic (e.g., "Video on Weatherproof pro on ACP bonding"), provide only the information directly relevant to the user query. Exclude any additional details beyond what is specifically asked by the user, ensuring the response focuses solely on the queried product or topic, such as ACP bonding. Additionally, always include the timestamp alongside the file name, whether in the response or in citation. Do not include information about other aspects of the video that are not directly relevant to the query asked.
Instruction 17: Strictly follow all the above instructions.
Follow the above instructions carefully and answer the user's question"""
##################################Chat local analytics methods#########################################

@retry(Exception, tries=10, delay=1)
def topic_extraction(input_text):
    try:
        value = random.randint(0, 1)
        AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
        openai_client = openai.AzureOpenAI(
            azure_endpoint = openai_api_base[value],
            api_version="2023-09-01-preview",
            api_key = openai_key[value]
        )
        messages = [{"role": "user", "content": f"Analyze the following text and extract the main topic discussed in the following text : {input_text} Keep the extracted topic short and make sure it is relevant to the text and does not exceed more than three words.\nTopic:"}]
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=messages,
            temperature=0,
            timeout=60
        )
        text = [response.choices[0].message.content.strip() for i in range(2)]
        return text
    except Exception as error:
        print(f"Failed->topic_extraction. Error {error}")
        return []

@retry(Exception, tries=10, delay=1)
def summary_generation(input_text):
    try:
        value = random.randint(0, 1)
        AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
        openai_client = openai.AzureOpenAI(
            azure_endpoint = openai_api_base[value],
            api_version="2023-09-01-preview",
            api_key = openai_key[value]
        )
        messages = [{"role": "user", "content": f"Please provide a complete summary of what the following document talks about in atmost one paragraph with consistent summary and concise information:\n\n{input_text}\n\nSummary:\n-"}]
        response = openai_client.chat.completions.create(
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

def remove_numbering(item):
    # Use regular expression to match and remove numbering if it exists
    return re.sub(r'^\d+\.\s+', '', item)

def sample_questions(prompt):
    try:
        value = random.randint(0, 1)
        AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
        openai_client = openai.AzureOpenAI(
            azure_endpoint = openai_api_base[value],
            api_version="2023-09-01-preview",
            api_key = openai_key[value]
        )
        messages = [{"role": "user", "content": f"Please provide 3 insightful sample questions using the following text, keep the questions relevant and concise.:\n\n{prompt}\nAvoid using any numbering in the questions.\nQuestions:\n-"}]
        response = openai_client.chat.completions.create(
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
        print(f"Failed->sample_questions. Error {error}")

def num_tokens_from_text(messages):
    encoding= tiktoken.get_encoding("cl100k_base")  
    num_tokens = 0
    num_tokens += len(encoding.encode(messages))
    return num_tokens

def get_analytics(id,flow,workspace):
    if flow == 'local':
        db = Database('ChatDocument',workspace)
    else:
        db = Database('DocumentCollection',workspace)
    status,ppt_meta = db.fetch_one_record({"_id":id},{'keywords':1,'summary':1,'sample_questions':1})
    if status:
        if flow=='local':
            return {
            "summary":ppt_meta['summary'],
            "keywords":ppt_meta['keywords'],
            "sample_questions":ppt_meta['sample_questions']
            }
        summary = ppt_meta['summary']
        return {
            "summary":summary,
            "keywords":ppt_meta['keywords'][:20],
            "sample_questions":ppt_meta['sample_questions']
        }
    return {
            "summary":"",
            "keywords":[],
            "sample_questions":[]
        }

################################### Chat cache ##########################################

def query_cache(search_query,filter,workspace):
    filter = json.dumps({'user_id':filter})
    search_embedding = get_embedding(search_query)
    result = get_search_result(CACHE_TABLE, search_embedding, CACHE_TABLE_COLUMNS, filter,workspace,search_query, True)
    print(result)
    if 'result' not in result:
        return False
    data_array = result['result']['data_array'] if 'data_array' in result['result'] else []
    # if result:
    if len(data_array):
        cache_result = data_array[0]
        id = cache_result[0]
        score = cache_result[5]*100
        print(score)
        if score>=95:
            db=Database("ChatCollection",workspace)
            chatid,idx = id.split("_")
            idx = int(idx)
            status,record = db.fetch_one_record({"_id":ObjectId(chatid)},{'chat_conversations':1,'citations':1})
            if status:
                print(record['chat_conversations'][idx+1])
                return (record['chat_conversations'][idx+1]['content'],record['citations'][idx+1])
            else:
                return False
    return False


###############################Local file Chat flow methods#####################################

def upload_userfile(url,display_name,workspace):
    file_meta = {}
    db = Database('ChatDocument',workspace)
    file_meta['file_path'] = url
    file_meta['file_name'] = os.path.basename(url)
    file_meta['display_name'] = display_name.split(".")[0]
    file_meta['file_type'] = file_meta['file_name'].split(".")[-1]
    status,id = db.insert_single_record(file_meta)
    if status:
        return status,id
    return status,id

def generate_preview(url,workspace):
    ext = url.split(".")[-1]
    if ext in ['ppt','pptx']:
        images,thumbnails = ppt2img_thumb(url,'local-docs')
    elif ext == 'pdf':
        images,thumbnails = pdf2img_thumb(url,'local-docs')
    file_meta = {}
    db = Database('ChatDocument',workspace)
    status,temp = db.fetch_one_record({'file_path':url})
    file_meta['file_preview'] = [{"chunk_number":num+1,"image_path":img} for num,(thumb,img) in enumerate(zip(thumbnails,images))]
    status,message = db.update_one_record(temp['_id'],file_meta)
    if status:
        return status,temp['_id']
    return status,message

def get_localfile_df(id,workspace):
    db = Database('ChatDocument',workspace)
    query = [ 
    {'$match': {'_id': id}},
    {'$project': {'display_name':1, 'file_preview':1}}, 
    {'$unwind': {'path': '$file_preview'}}, 
    {'$project': {'_id': 1, 'display_name':1,'chunk_number': '$file_preview.chunk_number', 'image_path': '$file_preview.image_path'}}
    ]
    status,data = db.fetch_aggregate(query)

    if status:
        files_df = pd.DataFrame(data)
        return files_df
    return pd.DataFrame()

def insert_text_db(id,data,file_type,workspace):
    db = Database('ChatDocument',workspace)
    contents = []
    full_text = ""
    if file_type == 'pdf':
        for i, (pagenum,_, content) in enumerate(data):
            content = re.sub(r'\s+', ' ', content)
            section = {
                "chunk_number": str(pagenum),
                "chunk_raw_text": content
            }
            full_text+=content
            contents.append(section)
    elif file_type in ['ppt','pptx']:
        for i, (content, pagenum) in enumerate(data):
            content = re.sub(r'\s+', ' ', content)
            section = {
                "chunk_number": str(pagenum),
                "chunk_raw_text": content
            }
            full_text+=content
            contents.append(section)
    ppt_meta = {}
    ppt_meta['content'] = contents
    ppt_meta['file_raw_text'] = full_text
    ppt_meta['modified_on'] = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    status, message = db.update_one_record(id,ppt_meta)
    return  status, message


def read_local_file(id,workspace):
    db_central=Database("workspace","central")
    db=Database("ChatDocument",workspace)
    job_id = None
    job_id = '1084737292698536'
    if job_id:
        workspaceurl=DATABRICKS_URL
        databricks_api_url = f"{workspaceurl}/api/2.1/jobs/run-now"
        token = PAT_TOKEN
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        payload = {
            "job_id": job_id,
            "job_parameters": {"type":"insert","doc_id":str(id), "workspace":workspace}
        }
 
        logging.info("Sending request to Databricks...")
        response = requests.post(databricks_api_url, headers=headers, json=payload)
        if response.status_code == 200:
            try:
                notebook_response = response.json()
                run_id = notebook_response.get("run_id")
                status, message = db.update_one_record(id, {"run_id":run_id})
                url = f"{workspaceurl}/api/2.1/jobs/runs/get?run_id={run_id}"
                payload = ""
                headers = {"Authorization": f"Bearer {token}"}
        
                response = requests.request("GET", url, headers=headers, data=payload)
                if response.status_code==200:
                    notebook_response = response.json()
                    try:
                        task_id= list(filter(lambda x: x["task_key"] == "File_Process_1", notebook_response["tasks"]))[0]['run_id']
                    except Exception as e:
                        print(e)
                    url = f"{workspaceurl}/api/2.1/jobs/runs/get-output?run_id={task_id}"
                    status = notebook_response["state"]["life_cycle_state"]
       
                    while status=="RUNNING" or status=="PENDING" or status=="BLOCKED":
                        response = requests.request("GET", url, headers=headers, data=payload)
                        if response.status_code==200:
                            notebook_response = response.json()
                            status = notebook_response['metadata']["state"]["life_cycle_state"]
 
                        else:
                            print(json.dumps({"notebook_response": "Error in fetching id"}))
                        if status=="RUNNING" or status=="PENDING" or status=="BLOCKED":
                            time.sleep(5)
                    data=notebook_response
                    job_status=notebook_response['metadata']["state"]["result_state"]
                    if job_status=='SUCCESS':
                        return True, "Successfully processed file"
                    else:
                        return False, "Error In File Process"
                return False, "Unable to process file"
            except Exception as e:
                return False, f"Error: full_process->{e}"
    else:
        return False, "Job Not Available"
                

def create_local_chat(id,display_name,source_path,user_id,workspace):
    try:
        now = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        db = Database('ChatCollection',workspace)
        # llm_citations = False
        chat_meta = {}
        chat_conversations = [{"role" : SYSTEM, "content" : system_message_chat_conversation}]
        citation = [[]]
        analytics = [{
            "summary":"",
            "keywords":[],
            "sample_questions":[]
        }]
        feedback = [{"feedback":None}]
        dates=[{"query_date": None,"feedback_date": None}]
        chat_conversations.append({"role" : ASSISTANT, "content" : ""})
        citation.append([])
        feedback.append({"feedback":2,
                         "assessment": None,
                        "feedback_description": ""})
        chat_meta['chat_flow'] = 'local'
        analysis = get_analytics(id,chat_meta['chat_flow'],workspace)
        analytics.append(analysis)
        chat_meta['user_id'] = user_id
        chat_meta['created_on'] = now
        chat_meta['chat_conversations'] = chat_conversations
        chat_meta['citations'] = citation
        chat_meta['analytics'] = analytics
        chat_meta['feedback'] = feedback
        chat_meta['dates'] = dates
        chat_meta['chat_file'] = display_name.split(".")[0]
        chat_meta['chat_file_id'] = id
        chat_meta['source_path'] = source_path
        chat_meta["title"] = chat_title([analysis['summary']])
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

def local_flow_process(url,display_name,user_id,workspace):#add userid
    status,id = upload_userfile(url,display_name,workspace)
    message = id if not status else ""
    if status:
        status,message = read_local_file(id,workspace)
        if status:
            status,chat_id = create_local_chat(id,display_name,url,user_id,workspace)
            if status:
                db = Database('ChatCollection',workspace)
                status,record = db.fetch_one_record({"_id":chat_id},{'title':1,'updated_on':1,'chat_flow':1})
                record['chat_id'] = record.pop('_id')
                return status,record
    return status, message
        

def local_flow_preview(url,workspace, display_name):#add userid
    while True:
        db=Database("ChatDocument",workspace)
        status,records=db.fetch_one_record({"display_name":display_name},{"_id":0,"run_id":1})
        if status and "run_id" in records:
            wokspaceurl=DATABRICKS_URL
            token = PAT_TOKEN
            
            run_id = records.get("run_id")
            
            url = f"{wokspaceurl}/api/2.1/jobs/runs/get?run_id={run_id}"
            payload = ""
            headers = {"Authorization": f"Bearer {token}"}

            response = requests.request("GET", url, headers=headers, data=payload)
            if response.status_code==200:
                notebook_response = response.json()
                try:
                    task_id= list(filter(lambda x: x["task_key"] == "File_Preview", notebook_response["tasks"]))[0]['run_id']
                except Exception as e:
                    print(e)
                url = f"{wokspaceurl}/api/2.1/jobs/runs/get-output?run_id={task_id}"
                status = notebook_response["state"]["life_cycle_state"]

                while status=="RUNNING" or status=="PENDING" or status=="BLOCKED":
                    response = requests.request("GET", url, headers=headers, data=payload)
                    if response.status_code==200:
                        notebook_response = response.json()
                        status = notebook_response['metadata']["state"]["life_cycle_state"]

                    else:
                        print(json.dumps({"notebook_response": "Error in fetching id"}))
                    if status=="RUNNING" or status=="PENDING" or status=="BLOCKED":
                        time.sleep(5)
                data=notebook_response
                # print(data)
                job_status=notebook_response['metadata']["state"]["result_state"]
                if job_status=='SUCCESS':
                    return True, "Successfully processed file"
                else:
                    return False, "Error In File preview"
            return False, "Unable to process file"
        else:
            time.sleep(2)
            continue

def local_index_update(display_name, workspace):
    # while True:
    db=Database("ChatDocument",workspace)
    status,records=db.fetch_one_record({"display_name":display_name},{"_id":0,"run_id":1})
    if status and "run_id" in records:
        wokspaceurl=DATABRICKS_URL
        token = PAT_TOKEN
        
        run_id = records.get("run_id")
        
        url = f"{wokspaceurl}/api/2.1/jobs/runs/get?run_id={run_id}"
        payload = ""
        headers = {"Authorization": f"Bearer {token}"}

        response = requests.request("GET", url, headers=headers, data=payload)

        if response.status_code==200:
            notebook_response = response.json()
            try:
                task_id= list(filter(lambda x: x["task_key"] == "Index_Update", notebook_response["tasks"]))[0]['run_id']
            except Exception as e:
                print(e)
            url = f"{wokspaceurl}/api/2.1/jobs/runs/get-output?run_id={task_id}"
            status = notebook_response["state"]["life_cycle_state"]
            while status=="RUNNING" or status=="PENDING" or status=="BLOCKED":
                response = requests.request("GET", url, headers=headers, data=payload)
                if response.status_code==200:
                    notebook_response = response.json()
                    status = notebook_response['metadata']["state"]["life_cycle_state"]

                else:
                    print(json.dumps({"notebook_response": "Error in fetching id"}))
                if status=="RUNNING" or status=="PENDING" or status=="BLOCKED":
                    time.sleep(5)
            data=notebook_response
            job_status=notebook_response['metadata']["state"]["result_state"]
            if job_status=='SUCCESS':
                return True, "Successfully updated index"
            else:
                return False, "Error In Index Update"
        return False, "Unable to index file"
    else:
        time.sleep(2)
        # continue

###############################Chat QA specific flow methods#####################################
def create_specific_chat(id,display_name,source_path,user_id,workspace):
    try:
        now = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        db = Database('ChatCollection',workspace)
        chat_meta = {}
        chat_conversations = [{"role" : SYSTEM, "content" : system_message_chat_conversation}]
        citation = [[]]
        analytics = [{
            "summary":"",
            "keywords":[],
            "sample_questions":[]
        }]
        feedback = [{"feedback":None}]
        dates=[{"query_date": None,"feedback_date": None}]
        chat_conversations.append({"role" : ASSISTANT, "content" : ""})
        citation.append([])
        feedback.append({"feedback":2,
                        "assessment": None,
                        "feedback_description": ""})
        chat_meta['chat_flow'] = 'specific'
        analysis = get_analytics(id,chat_meta['chat_flow'],workspace)
        analytics.append(analysis)
        chat_meta['user_id'] = user_id
        chat_meta['created_on'] = now
        chat_meta['chat_conversations'] = chat_conversations
        chat_meta['citations'] = citation
        chat_meta['analytics'] = analytics
        chat_meta['feedback'] = feedback
        chat_meta['dates'] = dates
        chat_meta['chat_file'] = display_name
        chat_meta['chat_file_id'] = id
        chat_meta['source_path'] = source_path
        chat_meta["title"] = chat_title([analysis['summary']])
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
    
def specific_flow_process(id,user_id,workspace):#add userid
    db = Database('DocumentCollection',workspace)
    status,record = db.fetch_one_record({"_id":id},{'_id':0,'display_name':1,'file_path':1})
    if status:
        status,chat_id = create_specific_chat(id,record['display_name'],record['file_path'],user_id,workspace)
        if status:
            db = Database('ChatCollection',workspace)
            status,record = db.fetch_one_record({"_id":chat_id},{'title':1,'updated_on':1,'chat_flow':1})
            record['chat_id'] = record.pop('_id')
            return status,record
    return status, record


##################################Chat QA chat methods#########################################

def modify_citations(text, names):
    escaped_names = [re.escape(name) for name in names]  # Escape special characters
    names_pattern = re.compile(r'(' + '|'.join(escaped_names) + r')', re.IGNORECASE)
    stripped_text = re.sub(r'[<>]{2,4}', '', text)
    modified_text = names_pattern.sub(r'<<<<\1>>>>', stripped_text)
    return modified_text

def num_tokens_from_messages(messages):
    encoding= tiktoken.get_encoding("cl100k_base")  
    num_tokens = 0
    for message in messages:
        num_tokens += 4  
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  
                num_tokens += -1  
    num_tokens += 2  
    return num_tokens

def get_embedding(text):
    url = f"{DATABRICKS_URL}/serving-endpoints/databricks-bge-large-en/invocations"

    payload = json.dumps({
    "input": [text]
    })
    headers = {
    'Authorization': f'Bearer {PAT_TOKEN}',
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print('Embedding: ', response)
    embedding_text = response.json()['data'][0]['embedding']
    return embedding_text

def get_search_result(table, query, clmns, filter, workspace, search_query, cache=False):
    workspace = workspace.replace('-','_')
    vs_ep_name = "dev_workspace"
    url = f"{DATABRICKS_URL}/api/2.0/vector-search/indexes/{CATALOG}.{workspace}.{table}_index/query"
    filter_dict = {}
    if "video" in search_query:
        filter_dict["file_type"] = "video"
    if isinstance(filter, str):
        filter_dict["display_name"] = filter
    if isinstance(filter, list) and len(filter)!=0:
        payload = {
        "query_text":search_query,
        "query_vector": query,
        "columns": clmns,
        "num_results": 30 if not cache else 1,
        "filters_json": json.dumps(filter_dict),
        "query_type":"HYBRID"
        }
    else:
        payload = {
        "query_text":search_query,
        "query_vector": query,
        "columns": clmns,
        "num_results": int(client.get_secret('NO-OF-DOCS').value) if not cache else 1,
        "filters_json": json.dumps(filter_dict) if not cache else filter,
        "query_type":"HYBRID"
        }

    headers = {
    'Authorization': f'Bearer {PAT_TOKEN}',
    'Content-Type': 'application/json'
    }
    print(f"Filter: {payload['filters_json']}")
    response = requests.request("GET", url, headers=headers, data=json.dumps(payload))
    print('Fetch Search Result ', response)
    return response.json()

def doc_search(search_query, filter, flow, workspace) -> str:
    search_query = re.sub(r'[-+&|!(){}[\]^"~*?:\\/]', lambda x: '\\' + x.group(), search_query)
    query_vect = get_embedding(search_query)
    if flow=='local':
        res = get_search_result(LOCAL_CHAT_TABLE, query_vect, LOCAL_CHAT_TABLE_COLUMNS, filter, workspace, search_query)
    else:
        res = get_search_result(MASTER_TABLE, query_vect, MASTER_TABLE_COLUMNS, filter, workspace, search_query)
    r = []
    if 'data_array' not in res['result']:
        return [],""
    if isinstance(filter, list):
        data_array = []
        for result in res['result']['data_array']:
            if result[4] not in filter:
                data_array.append(result)
    else:
        data_array = res['result']['data_array']    
    for result in data_array[:10]:
        temp = {
                'file_name':result[5],
                'image_path': result[7],
                'source_path':result[6],
                'page':result[3],
                'file': result[4],
                'content':result[2],
                'id':result[1],
                'modified_on':result[8]
            }
        r.append(temp)

    results = []
    for doc in r:
        result = doc["file"].title() + ": " + doc['modified_on'] + ": " + doc['content'].replace("\n", "").replace("\r", "")
        results.append(result)

    content = "\n".join(results)
    return r, content

def repo_search(workspace,search_query, filter=None) -> str:
    print("Repo Search")
    query_vect = get_embedding(search_query)
    res = get_search_result(REPO_TABLE, query_vect, REPO_TABLE_COLUMNS, filter, workspace, search_query)
    results = []
    if 'data_array' not in res['result']:
        return pd.DataFrame(),""
    if isinstance(filter, list):
        data_array = []
        for result in res['result']['data_array']:
            if os.path.splitext(result[2])[0] not in filter:
                data_array.append(result)
    else:
        data_array = res['result']['data_array']
    for result in data_array[:10]:
        temp_content = {
            'file_name': f'{result[2]}',
            'summary': result[3],
            'file_type': result[4],
            'file_path': result[5],
            'modified_on': result[6]
            }
        results.append(temp_content)
    results = sorted(results, key=lambda x: x['modified_on'], reverse=True)
    content = [doc["file_name"] + ": " + doc['summary'] for doc in results]
    content = "\n".join(content)
    if results:
        return pd.DataFrame(results)[['file_name','file_path']],content
    else:
        return pd.DataFrame(),content
    
@retry(Exception, tries=10, delay=2)
def stream_repo_response(messages):
    try:
        base = client.get_secret("AZURE-OPENAI-ENDPOINT-US").value
        key = client.get_secret('OPENAI-API-KEY-US').value
        model = client.get_secret("CHAT-MODEL").value
        openai_api_base = [base,base]
        openai_key = [key,key]
        DEPLOYMENT_NAME = [model,model]
        value = random.randint(0, 1)
        AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
        openai_client = openai.AzureOpenAI(
            azure_endpoint = openai_api_base[value],
            api_version="2023-09-01-preview",
            api_key = openai_key[value]

        )
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=messages,
            top_p = 0,
            max_tokens=1024,
            timeout=100,
            stream = True
            )
        for chunk in response:
            if chunk is not None and len(chunk.choices):
                yield True,chunk.choices[0].delta.content if chunk.choices[0].delta.content!=None else ""
            else:
                yield False,""
    except Exception as error:
        print(f"Failed->repo_response. Error {error}")
        yield False,""


def time_to_seconds(time_string):
    # Extract minutes and seconds using regular expression
    match = re.match(r'(\d+):(\d+)', time_string)
    
    if match:
        minutes, seconds = map(int, match.groups())
        total_seconds = minutes * 60 + seconds
        return total_seconds
    else:
        return None

def get_citations(assistant_response,documents,llm_citations,chat_flow,repo=False):
    pattern = r'\<<<<(.*?)\>>>>'
    citations = set(re.findall(pattern, assistant_response))
    c_list=[]
    if citations:
        for i, citation in enumerate(citations, 1):
            c_dict={}
            c_dict['citation_num']=str(i) if chat_flow == 'general' else documents[documents['file']==citation]['page'].iloc[0]
            if llm_citations and len(citation.split(",",1))>1:
                name,url = citation.split(",",1)
                c_dict['name'] = name
                c_dict['url'] = url.strip()
                time_match = re.search(r'(\d+:\d+)', name)
                if 'mp4' in name and time_match:
                    time_string = time_match.group(1)
                    c_dict['citation_type'] = 'video'
                    c_dict['duration'] = time_to_seconds(time_string)
                elif 'mp3' in name and time_match:
                    time_string = time_match.group(1)
                    c_dict['citation_type'] = 'audio'
                    c_dict['duration'] = time_to_seconds(time_string)
                else:
                    c_dict['citation_type'] = 'general'
            else:
                c_dict['name'] = citation
                if repo:
                    c_dict['url'] = documents[documents['file_name']==citation]['file_path'].iloc[0]
                    time_match = re.search(r'(\d+:\d+)', citation)
                    if 'mp4' in citation and time_match:
                        time_string = time_match.group(1)
                        c_dict['citation_type'] = 'video'
                        c_dict['duration'] = time_to_seconds(time_string)
                    elif 'mp3' in citation and time_match:
                        time_string = time_match.group(1)
                        c_dict['citation_type'] = 'audio'
                        c_dict['duration'] = time_to_seconds(time_string)
                    else:
                        c_dict['citation_type'] = 'general'
                else:
                    c_dict['image_path'] = documents[documents['file']==citation]['image_path'].iloc[0]
                    c_dict['source_path'] = documents[documents['file']==citation]['source_path'].iloc[0]
                    c_dict['citation_type'] = 'kb'
            c_list.append(c_dict)
    return c_list

def llm_kb(search_query):
    value = random.randint(0, 1)
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
    openai.azure_endpoint = openai_api_base[value]
    openai.api_key = openai_key[value]
    system_msg = f"""
    You are a helpful web assistant. You can answer questions using web sources. Must follow all the instructions below on how to behave without fail.
    Instruction 1: Answer the question below by searching the web source. 
    Instruction 2: Use quadruple angle brackets to include domain name of URL and exact web source URL like <<<<DOMAIN NAME OF URL, WEB URL>>>> in the response e.g. <<<<wikipedia, https://www.wikipedia.org>>>> based on user question. Do not include square brackets '[]' or open brackets '()' within quadruple angle brackets '<<<<>>>>'.
    Instruction 3: If response coming from multiple web sources, include all the web source URLs in the response where ever content mentioned.
    Instruction 4: Please make sure that the given web URL is valid and working. Don't include invalid web URL in the response.
    Instruction 5: Make sure that the response should always have web source reference. Don't give response without web source reference.
    Instruction 6: If query asks about you, please answer with the exact text delimited by double quotes "I'm GPTXponent, your companion for document analysis, information extraction and insights generation. I can quickly process and understand text-based content, helping you extract key information, identify trends, and generate valuable insights from your documents. Just upload your documents or ask me questions, and I'll assist you in making your document-related tasks more efficient and insightful.".
    """
    # msg = [{"role": SYSTEM, "content": system_msg}, {"role": USER, "content": search_query}]
    msg = [{"role": USER, "content": search_query}]
    completion = openai.chat.completions.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=msg,
        temperature=0.5,
        max_tokens=1024,
        timeout=20)
    result = completion.choices[0].message.content
    return result

def parse_citations(chat_history,citations,feedbacks,workspace):
    for chat,citation in zip(chat_history,citations):
        chat['citations'] = citation
        if citation:
            for cite in citation:
                if cite['citation_type'] in ['kb','video','audio','general']:
                    chat['content'] = chat['content'].replace(f"<<<<{cite['name']}>>>>", f"<<<<{cite['citation_num']}>>>>")
                    if cite['citation_type'] in ['video','audio']:
                        cite['url'] = generate_link(cite['url'],workspace) if "azureexponentiaai.sharepoint.com" in cite['url'] else cite['url']
                else:
                    chat['content'] = chat['content'].replace(f"<<<<{cite['name']}, {cite['url']}>>>>", f"<<<<{cite['citation_num']}>>>>")
    chat_output = []
    # print(citations)
    for chat,citation,feedback in zip(chat_history,citations,feedbacks):
        chats = {"role":chat['role']}
        chats['data'] = []
        if citation:
            chats['citations'] = citation
            chats.update(feedback)
            # chats['feedback'] = feedback
            citation = pd.DataFrame(citation)
            pattern = r'(?<=\>>>>)(?=[\s+\n+\r+a-zA-Z0-9:\"\'.])'
            points = re.split(pattern, chat['content'])
            for point in points:
                point=point.lstrip(".")
                chat_contents = {}
                pattern = r'\<<<<(.*?)\>>>>'
                cits = re.findall(pattern, point)
                pattern = r'\<<<<\d+\>>>>'
                chat_contents['content'] = re.sub(pattern, '', point).rstrip("\n")
                chat_contents['citations'] = []
                for cit in cits:
                    # print(citation[citation['citation_num']==cit].to_dict('records'))
                    chat_contents['citations'].append(citation[citation['citation_num']==cit].to_dict('records')[0])
                chats['data'].append(chat_contents)
        else:
            chats['data'].append({'content':chat['content']})
            chats['citations'] = []
            chats.update(feedback)
        chat_output.append(chats)
    return chat_output

@retry(Exception, tries=10, delay=1)
def chat_title(chat_conversations):
    value = random.randint(0, 1)
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
    openai_client = openai.AzureOpenAI(
            azure_endpoint = openai_api_base[value],
            api_version="2023-09-01-preview",
            api_key = openai_key[value]

        )
    message = [{"role" : USER, "content" : f"Conversation start:\n{str(chat_conversations[-2:])}]=\nConversation end.\nSummarize the conversation in 5 words or fewer:"}]
    chat_completion = openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=message, 
        temperature=0, 
        max_tokens=35, 
        timeout=10)
    chat_content = chat_completion.choices[0].message.content.strip()
    return chat_content

@retry(Exception, tries=10, delay=1)
def query_generation(query,chat_conversations):
    base = client.get_secret("AZURE-OPENAI-ENDPOINT-US").value
    key = client.get_secret('OPENAI-API-KEY-US').value
    model = client.get_secret("CHAT-MODEL").value
    openai_api_base = [base,base]
    openai_key = [key,key]
    DEPLOYMENT_NAME = [model,model]
    value = random.randint(0, 1)
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
    openai_client = openai.AzureOpenAI(
        azure_endpoint = openai_api_base[value],
        api_version="2023-09-01-preview",
        api_key = openai_key[value]
    )

    mod_chat_history = chat_conversations.copy()
    del mod_chat_history[0]
    
       
    task = """
        Identify if the Current User Query is independent or dependent.
        If you think the Current User Query is independent and does not require any completion simply return the Current User Query as the Modified Query.
        If you think the Current user query is dependent on previous conversation then only and only then try to remove this dependency from the Current User Query by taking relevant context from the Chat History and return the independent query as the Modified Query.
        Donot remove mentions of any files/audios in the original query. Keep them in the modified query as well.
    """
    msg = [{"role": "user", "content": f"Chat History:{mod_chat_history[-4:]}\nCurrent User Query: {query}\nSystem Task:{task}\nModified Query (if necessary):\n"}]
    completion = openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=msg,
        temperature=0,
        max_tokens=100,
        timeout=20)
    result = completion.choices[0].message.content
    return result

def local_general_responses(chat_conversations):
    value = random.randint(0, 1)
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
    openai_client = openai.AzureOpenAI(
            azure_endpoint = openai_api_base[value],
            api_version="2023-09-01-preview",
            api_key = openai_key[value]

        )
    system_msg = f"""
    You are a helpful assistant. Follow the below instructions to help users:
    Instruction 1: If the user's query is affirmative (e.g., "yes," "okay," "that's fine"), respond with: Great !! How can I help you further ?
    Instruction 2: If the user's query is an exit phrase (e.g., "bye," "goodbye," "see you"), respond with: Good to have a nice conversation with you .... 
    Instruction 3: If the user asks a question related to a previous question or topic in the conversation, respond with the last user message.
    
    """
    chat_conversations[0]['content'] = system_msg
    completion = openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=chat_conversations,
        temperature=0.5,
        max_tokens=1024,
        timeout=20)
    result = completion.choices[0].message.content
    return result

def retrieve_chat(chat_id,workspace):
    try:
        db = Database('ChatCollection',workspace)
        status,record = db.fetch_one_record({'_id':chat_id})
        chat_meta = record
        chat_conversations = chat_meta['chat_conversations']
        citation = chat_meta['citations']
        feedbacks = chat_meta['feedback']
        chat_meta['chat_conversations'] = parse_citations(chat_conversations[-1:],citation[-1:],feedbacks[-1:],workspace)
        for chat, analytics in zip(chat_meta['chat_conversations'], chat_meta['analytics'][-1:]):
            chat.update(analytics)
        chat_meta.pop('citations', None)
        chat_meta.pop('analytics', None)
        chat_meta.pop('feedback', None)
        chat_meta['chat_id'] = chat_meta.pop('_id')
        return True,chat_meta
    except Exception as error:
        print(f'Failed chat_utils->retrieve_chat. Error:{error}')
        return False,"Failed to retrieve chat."

def retrieve_chats(chat_id,workspace):
    try:
        db = Database('ChatCollection',workspace)
        status,record = db.fetch_one_record({'_id':chat_id})
        chat_meta = record
        chat_conversations = chat_meta['chat_conversations']
        citation = chat_meta['citations']
        feedbacks = chat_meta['feedback']
        chat_meta['chat_conversations'] = parse_citations(chat_conversations,citation,feedbacks,workspace)[1:]
        for chat, analytics in zip(chat_meta['chat_conversations'], chat_meta['analytics'][1:]):
            chat.update(analytics)
        chat_meta.pop('citations', None)
        chat_meta.pop('analytics', None)
        chat_meta.pop('feedback', None)
        chat_meta['chat_id'] = chat_meta.pop('_id')
        return True,chat_meta
    except Exception as error:
        print(f'Failed chat_utils->retrieve_chat. Error:{error}')
        return False,"Failed to retrieve chat."

@retry(Exception, tries=10, delay=1)
def stream_kb_response(chat_conversations):
    try:
        base = client.get_secret("AZURE-OPENAI-ENDPOINT-US").value
        key = client.get_secret('OPENAI-API-KEY-US').value
        model = client.get_secret("CHAT-MODEL").value
        openai_api_base = [base,base]
        openai_key = [key,key]
        DEPLOYMENT_NAME = [model,model]
        value = random.randint(0, 1)
        AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
        openai_client = openai.AzureOpenAI(
            azure_endpoint = openai_api_base[value],
            api_version="2023-09-01-preview",
            api_key = openai_key[value]
        )

        chat_completion = openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT, 
        messages=chat_conversations,
        # temperature=0.7,
        top_p =0,
        max_tokens=max_response_tokens,
        stream = True
        )
        print(chat_completion)
        for chunk in chat_completion:
            if chunk is not None and len(chunk.choices):
                # time.sleep(0.1)
                yield True,chunk.choices[0].delta.content if chunk.choices[0].delta.content!=None else ""
                
    except Exception as error:
        print(f'Failed:kb_response. Error: {error}')
        yield False,""

@retry(Exception, tries=10, delay=1)
def follow_up(input_text):
    try:
        value = 0
        AZURE_OPENAI_CHATGPT_DEPLOYMENT = DEPLOYMENT_NAME[value]
        openai_client = openai.AzureOpenAI(
            azure_endpoint = openai_api_base[value],
            api_version="2023-09-01-preview",
            api_key = openai_key[value]

        )
        messages = [{"role": "user", "content": f"""Extract the subject from the query and combine it with -> 
                     'The question seems a bit unclear. Could you please provide more context or details about the query subject you're referring to?'
                     Only in case of statements like 'hi','hello','ok',etc., give general response.
                     Return only the Response:\nQuery:{input_text}
                     Donot give 'Response:'              
                    """}]
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=messages,
            temperature=0,
            max_tokens=100,
            timeout=20
        )
        text = response.choices[0].message.content
        if text:
            return text.strip().strip('"').strip("'")
        return ""   
    except Exception as error:
        print(f"Failed->follow_up. Error {error}")

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

def generate_stream_response(query,user_id,chat_id,workspace):
    try:
        exclude_files = []
        db = Database('UserActivity',workspace)
        status,record = db.fetch_one_record({'user_id':user_id},{'chat_filter':1})
        # print(record)
        if status:
            if 'chat_filter' in record:
                for file in record['chat_filter']:
                    # file = file.replace("'","")
                    exclude_files.append(f"{file}")
        now = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        today_date=datetime.now().strftime('%Y-%m-%d')
        db = Database('ChatCollection',workspace)
        llm_citations = False
        chat_meta = {}
        filter = None
        if not chat_id:
            chat_conversations = [{"role" : SYSTEM, "content" : system_message_chat_conversation}]
            chat_meta['user_id'] = user_id
            chat_meta['created_on'] = now
            chat_meta['chat_conversations'] = []
            chat_meta['chat_flow'] = 'general'
            status,id = db.insert_single_record(chat_meta)
            message = id if not status else ""
            if not status:
                return False,message
            
            chat_id = id
            print('New Chat entry done.')
            citation = [[]]
            analytics = [{
                "summary":"",
                "keywords":[],
                "sample_questions":[]
                }]
            feedback = [{"feedback":None}]
            dates=[{"query_date": None,"feedback_date": None}]
        else:
            status,record = db.fetch_one_record({'_id':chat_id})
            chat_meta.update(record)
            chat_conversations = chat_meta['chat_conversations']
            citation = chat_meta['citations']
            analytics = chat_meta['analytics']
            feedback = chat_meta['feedback']
            if 'dates' in chat_meta:
                dates = chat_meta['dates']
            print('Chat history retrieved.')
        if chat_meta['chat_flow']=='specific' or chat_meta['chat_flow']=='local':
            filter = chat_meta['chat_file']
        temp_chat_history = copy.deepcopy(chat_conversations)
        conv_history_tokens = num_tokens_from_messages(temp_chat_history)

        while conv_history_tokens + max_response_tokens >= token_limit:
            if chat_meta['chat_flow']=='general':
                del temp_chat_history[1] 
            else:
                del temp_chat_history[1]
            conv_history_tokens = num_tokens_from_messages(temp_chat_history)

        if len(query.strip())>0:

            if chat_meta['chat_flow'] !='general' and len(chat_conversations)==2 and chat_conversations[1]['role']=='assistant':
                temp_chat_history[1]['content'] = analytics[1]['summary']
            
            search = query_generation(query,temp_chat_history)
            
            print("Original query for document search:",query)
            print("Generated query for KB search: ",search)

            if not search.strip():
                search = query

            citation.append([])
            analytics.append({
                "summary":"",
                "keywords":[],
                "sample_questions":[]
                })
            feedback.append({"feedback":None})
            if len(dates):
                dates.append({"query_date": today_date,"feedback_date": None})


            repo_pattern = re.compile(r'\b(video|videos|audio|audios|watch|play|listing|list)\b', re.IGNORECASE)

            temp_chat_history = temp_chat_history[:1]

            # cache_response = query_cache(search, user_id, workspace)
            cache_response=False
            print("Checked cache")

            if not cache_response or chat_meta['chat_flow']=='specific' or chat_meta['chat_flow']=='local':
                if re.search(repo_pattern, search.lower()) and chat_meta['chat_flow']=='general':
                    print("inside repo search")
                    filter = exclude_files
                    messages = []
                    llm_docs,sources = repo_search(workspace,search,filter)
                    if llm_docs.empty:
                        status, chat_content = (True, "I apologize, but it seems that you don't have access to that information. Is there something else I can assist you with?")
                        citations = []
                    else:
                        input_text = f"User Query:\n{search}\nSources:\n{sources}"
                        messages = [{"role": "system", "content": system_message_repo_conversation}]
                        messages.append({"role": "user", "content": input_text})
                        print(llm_docs)

                        chat_content = ''
                        for status,response in stream_repo_response(messages):
                            chat_content+=response
                            if status:
                                time.sleep(0.1)
                                yield 'stream',json.dumps({"text":response})
                        print("Repo response streaming completed")
                        chat_content = modify_citations(chat_content,list(llm_docs['file_name'].values))


                        citations = get_citations(chat_content,llm_docs,False,chat_meta['chat_flow'],True)
                        print("Extracted citations from repo index reponse.")
                else:
                    print("filter:",filter)
                    if chat_meta['chat_flow']=='general':
                        filter = exclude_files
                    r, content = doc_search(search, filter, chat_meta['chat_flow'], workspace)
                   
                    print("Retrieved relevant KB docs.")

                    for item in r:
                        item.pop("content", None)

                    kb_docs = pd.DataFrame(r)

                    for chats in temp_chat_history[1:]:
                        pattern = r'\<<<<(.*?)\>>>>'
                        chats['content'] = re.sub(pattern, '', chats['content'])

                    if len(kb_docs)>0:
                        kb_docs['file'] = kb_docs['file'].apply(lambda x: x.title())
                        print(kb_docs[['file_name','page']])

                    user_content = f"sources:{content} \n user question:{search}"
                    temp_chat_history.append({"role": USER, "content": user_content})
                
                    if kb_docs.empty:
                        status, chat_content = (True, "I apologize, but it seems that you don't have access to that information. Is there something else I can assist you with?")
                    else:
                        chat_content = ''
                        for status,response in stream_kb_response(temp_chat_history):
                            chat_content+=response
                            if status:
                                time.sleep(0.1)
                                yield 'stream',json.dumps({"text":response})
                    if not status:
                        raise Exception('Error in generating reponse please try again')
                    print("Generated user query response.")
                    print(chat_content)
                    if len(kb_docs)>0:
                        # print('BEFORE MODIFICATION: ', chat_content)
                        chat_content = modify_citations(chat_content,list(kb_docs['file'].values))
                        # print('AFTER MODIFICATION: ', chat_content)   
                        def replace_with_list1(match):
                            substring = match.group(1)
                            return next((f"<<<<{string1}>>>>" for string1 in list(kb_docs['file'].values) if substring in string1), substring)
                        pattern = r'\<<<<(.*?)\>>>>'
                        chat_content = re.sub(pattern, replace_with_list1, chat_content)

                    print("Generated user query response post citation modification.")

                    citations = get_citations(chat_content,kb_docs,llm_citations,chat_meta['chat_flow'])

                    print("Extracted citations.")
                    # print(citations)
            else:
                chat_content,citations = cache_response

            chat_conversations.append({"role":USER,"content" : query})
        else:
            chat_conversations.append({"role":USER,"content" : query})
            chat_content = "I apologize, but it seems like your message is empty or unclear. Please provide a valid question or input, and I'll be happy to assist you. If you need any help or have any inquiries, feel free to ask."
            citations = []
        chat_content = convert_pipe_to_html(chat_content)
        chat_conversations.append({"role":ASSISTANT, "content": chat_content})
        citation.append(citations)
        analytics.append({
                "summary":"",
                "keywords":[],
                "sample_questions":[]
                })
        feedback.append({"feedback":2,
                        "assessment": None,
                        "feedback_description": ""
                        })
        if len(dates):
            dates.append({"query_date": None,"feedback_date": None})

        print("Updated user query response.")


        if "title" not in chat_meta.keys():
            chat_meta["title"] = chat_title(chat_conversations[-2:])
            print("Generated Title for Conversation.")

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

        logs = {}
        logs['user_id'] = user_id
        logs['timestamp'] = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        logs['query'] = {"Original query":query,
                         "Modified query":search}
        logs['chat_id'] = chat_id if chat_id else None
        logs['status_code'] = 200
        logs['status_description'] = "Successfully generated response"
        logs['api_status'] = "Succeeded"
        chat_logs(logs,workspace)

        yield True,json_util.dumps(chat_meta)

    except Exception as error:
        print(f'Failed chat_utils->generate_response. Error:{error}')

        logs = {}
        logs['user_id'] = user_id
        logs['timestamp'] = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        logs['query'] = query
        logs['chat_id'] = chat_id if chat_id else None
        logs['status_code'] = 400
        logs['status_description'] = error
        logs['api_status'] = "Succeeded"
        chat_logs(logs,workspace)

        yield False,"Failed to retrieve chat/response"

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

import concurrent.futures

def delete_file_index(id,workspace):
    db = Database('ChatDocument',workspace)
    status,record = db.fetch_one_record({"_id":id})
    delete_blob(record['file_path'])
    # [delete_blob(preview['image_path']) for preview in record['file_preview']]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit delete_blob function for each image_path in record['file_preview']
        futures = [executor.submit(delete_blob, preview['image_path']) for preview in record['file_preview']]
        
        # Wait for all threads to complete
        concurrent.futures.wait(futures)
    #NOTE: Changed this
    # remove_from_index(record['display_name'],[local_index.format(workspace)])
    status = remove_from_index(id,workspace)
    if status:
        db.delete_single_record(id)
    return True

def delete_chat(chat_id,workspace):
    db = Database('ChatCollection',workspace)
    status,record = db.fetch_one_record(chat_id)
    if status:
        if record['chat_flow']=='local':
            delete_file_index(record['chat_file_id'],workspace)
    status,message = db.delete_single_record(chat_id)
    if status:
        return True,"Successfully deleted chat record."
    return False,message

def fetch_chat_history(user_id,workspace):
    db_central = Database('workspace','central')
    status,rec = db_central.fetch_one_record({'workspace_name':workspace},{'isIndexed':1})
    if status and 'isIndexed' in rec:
        local_chat_flag=rec['isIndexed']        
    else:
        print('no isIndexed in rec..')
        local_chat_flag=False
    db = Database('ChatCollection',workspace)
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
    return True, {"data":data,"local_chat_index":local_chat_flag}

def get_chat_preview(chat_id,workspace):
    db = Database('ChatCollection',workspace)
    status,record = db.fetch_one_record({'_id':chat_id})
    if status:
        file_id = record['chat_file_id']
        if record['chat_flow'] == 'local':
            db = Database('ChatDocument',workspace)
        else:
            db = Database('DocumentCollection',workspace)
        status,record = db.fetch_one_record({'_id':file_id},{'_id':0,'file_preview':1})
        if status:
            return True, record
    return False,record

def toggle_feedback(chat_id,value,index,workspace):
    db = Database('ChatCollection',workspace)
    status,record = db.fetch_one_record({'_id':chat_id},{'_id':0,'feedback':1})
    if status:
        feedback = record['feedback']
        feedback[index+1] = value
        status, message = db.update_one_record(chat_id,{"feedback":feedback})
        if status:
            return True,"Feedback Update Success."
        return True, message
    return False,"Feedback Update Failed."

def toggle_feedback_v2(chat_id, feedback, index, assessment, feedback_description, workspace):
    db = Database('ChatCollection',workspace)
    # print(workspace,chat_id)
    status,record = db.fetch_one_record({'_id':chat_id},{'_id':0,'feedback':1})
    if status:
        feedback_list = record['feedback']
        feedback_list[index+1]['feedback'] = feedback
        feedback_list[index+1]['assessment'] = assessment
        feedback_list[index+1]['feedback_description'] = feedback_description
        
        status, message = db.update_one_record(chat_id,{"feedback":feedback_list})
        if status:
            return True,"Feedback Update Success."
        return True, message
    return False,"Feedback Update Failed."

def chat_logs(logs,workspace):
    db = Database('ChatLogs',workspace)
    db.insert_single_record(logs)