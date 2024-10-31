import re
import time
import pandas as pd
from datetime import datetime
from common.database import Database
from common.chat_utils import (query_generation,
                               repo_search,
                               system_message_repo_conversation,
                               stream_repo_response,
                               stream_kb_response,
                               modify_citations,
                               time_to_seconds,
                               USER,
                               SYSTEM,
                               doc_search,
                               convert_pipe_to_html
                               )

system_message_chat_conversation = """ 
Welcome to GPTXponent, where we specialize in document analysis and information extraction. Please provide your query, and I will assist you by referencing our extensive knowledge base.
 
When formulating your response, adhere to the following guidelines:

Keep you answers as concise as possible with no additional information unrelated to the questions asked. Include all citations in the response you give.
 
If response contains any tables give it in a pipe separated table format.

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

def parse_citations_wa(chat,citations):
    if citations:
        for cite in citations:
            if cite['citation_type'] in ['kb','video','general']:
                new_cite = "{"+cite['citation_num']+"}"
                chat = chat.replace(f"<<<<{cite['name']}>>>>", new_cite)
    chat = chat+"\nSources:\n"+"\n".join(f"{cite['citation_num']}:{cite['source_path'] if cite['source_path'] else cite['url']}" for cite in citations)
    return chat

def get_citations_wa(assistant_response,documents,llm_citations,repo=False):
    pattern = r'\<<<<(.*?)\>>>>'
    citations = set(re.findall(pattern, assistant_response))
    c_list=[]
    if citations:
        for i, citation in enumerate(citations, 1):
            c_dict={}
            c_dict['citation_num']=str(i)
            c_dict['source_path'] = None
            if llm_citations or len(citation.split(",",1))>1:
                name,url = citation.split(",",1)
                c_dict['name'] = name
                c_dict['url'] = url.strip()
                time_match = re.search(r'(\d+:\d+)', name)
                if 'mp4' in name and time_match:
                    time_string = time_match.group(1)
                    c_dict['citation_type'] = 'video'
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
                    else:
                        c_dict['citation_type'] = 'general'
                else:
                    c_dict['image_path'] = documents[documents['file']==citation]['image_path'].iloc[0]
                    c_dict['source_path'] = documents[documents['file']==citation]['source_path'].iloc[0]
                    c_dict['citation_type'] = 'kb'
            c_list.append(c_dict)
    return c_list

def generate_wa_chat(query,phone_number):
    try:
        db =Database('users','central')
        status,user_data = db.fetch_one_record({"phone_number":phone_number},{'email':1,'workspace':1})
        if not status:
            return True,{"response":"User does not exist with AIXponent","phone_no":phone_number}
        user_id = user_data['email']
        workspace_name=user_data['workspace']
        now = datetime.now().strftime('%d-%m-%Y %H:%M:%S')

        db =Database('ChatCollection',workspace_name)
        chat_date = str(datetime.now().date())
        chat_meta = {}
        status,record = db.fetch_one_record({'user_id':user_id,'phone_number':phone_number,'created_date':chat_date},{})
        if status:
            chat_meta = record
            temp_chat_history = chat_meta['chat_conversation']
            chat_id = chat_meta['_id']
        else:
            temp_chat_history=[{"role": "system", "content": system_message_chat_conversation}]
            chat_meta['user_id'] = user_id
            chat_meta['phone_number'] = phone_number
            chat_meta['created_on'] = now
            chat_meta['created_date'] = chat_date
            chat_meta['chat_conversations'] = temp_chat_history
            chat_meta['chat_flow'] = 'whatsapp'
            status,id = db.insert_single_record(chat_meta)
            message = id if not status else ""
            if not status:
                return False,message
            chat_id = id
            print('New Chat entry done.')
            
        filter = []
        if query.strip():
            search = query_generation(query,temp_chat_history)
            repo_pattern = re.compile(r'\b(video|videos|watch|play|listing|list)\b', re.IGNORECASE)
            if re.search(repo_pattern, search.lower()):
                # filter = f"not search.in(file_name, '{'|'.join(exclude_files)}','|')"
                filter = []
                messages = []
                llm_docs,sources = repo_search(search,filter)
                if llm_docs.empty:
                    status, chat_content = (True, "Sorry, I don't find any relevant information in my knowledge base.")
                else:
                    input_text = f"User Query:\n{search}\nSources:\n{sources}"
                    messages = [{"role": "system", "content": system_message_repo_conversation}]
                    messages.append({
                        "role": "user",
                        "content": input_text
                    })
                    print(llm_docs)
                    chat_content = ''
                    for status,response in stream_repo_response(messages):
                        chat_content+=response
                        if status:
                            time.sleep(0.1)
                            # yield 'stream',json.dumps({"text":response})
                    print("Repo response streaming completed")
                    print(chat_content)
                    chat_content = modify_citations(chat_content,list(llm_docs['file_name'].values))
                    citations = get_citations_wa(chat_content,llm_docs,False,True)
            else:
                print("filter:",filter)
                r, content = doc_search(search, filter, flow="general",workspace=workspace_name)

                print("Retrieved relevant KB docs.")

                for item in r:
                    item.pop("content", None)

                kb_docs = pd.DataFrame(r)
                if len(kb_docs)>0:
                    kb_docs['file'] = kb_docs['file'].apply(lambda x: x.title())
                    print(kb_docs[['file_name','page']])

                user_content = f"sources:{content} \n user question:{search}"
                messages = temp_chat_history[:1]
                messages.append({
                    "role": USER, 
                    "content": f"""{user_content}\n
                    Give very concise responses when ever possible using as few words as possible unless asked by the user to elaborate the answer. This is the most important rule to be followed.
                    """
                })
                if kb_docs.empty:
                    status, chat_content = (True, "Sorry, I don't find any relevant information in my knowledge base.")
                else:
                    chat_content = ''
                    for status,response in stream_kb_response(messages):
                        chat_content+=response
                        if status:
                            time.sleep(0.1)
                            # yield 'stream',json.dumps({"text":response})
                if not status:
                    raise Exception('Error in generating reponse please try again')
                chat_content = modify_citations(chat_content,list(kb_docs['file'].values))
                

                def replace_with_list1(match):
                    substring = match.group(1)
                    return next((f"<<<<{string1}>>>>" for string1 in list(kb_docs['file'].values) if substring in string1), substring)
                pattern = r'\<<<<(.*?)\>>>>'
                chat_content = re.sub(pattern, replace_with_list1, chat_content)
                print("Generated user query response.")
                print(chat_content)
                citations = get_citations_wa(chat_content,kb_docs,False)
        else:
            chat_content = "I apologize, but it seems like your message is empty or unclear. Please provide a valid question or input, and I'll be happy to assist you. If you need any help or have any inquiries, feel free to ask."

        print("Updated user query response.")
        print(citations)

        chat_content = parse_citations_wa(chat_content,citations)
        # chat_content = convert_pipe_to_html(chat_content)
        temp_chat_history.append({
            "role": USER, 
            "content": query
        })
        temp_chat_history.append({
            "role": SYSTEM,
            "content":chat_content
        })
        print(chat_content)
        if "title" not in chat_meta.keys():
            chat_meta["title"] = f"Whatsapp chat session: {chat_date}"
        chat_meta['updated_on'] = now
        chat_meta['chat_conversation'] = temp_chat_history

        status,message = db.update_one_record(chat_id,chat_meta)
        
        return True,{"response":chat_content,"phone_no":phone_number}
    except Exception as error:
        print(f'Failed chat_utils->generate_response. Error:{error}')
        return False,"Failed to retrieve chat/response"

