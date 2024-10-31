import pandas as pd
import openai
import concurrent.futures

from common.utils import num_tokens_from_text
from common.keyvault_connection import get_conn

client=get_conn()

#OpenAI
openai_key = client.get_secret('OPENAI-API-KEY-SC').value
openai_api_base = client.get_secret("AZURE-OPENAI-ENDPOINT-SC").value
DEPLOYMENT_NAME = "gpt-4o"
openai_client = openai.AzureOpenAI(
    azure_endpoint = openai_api_base,
    api_version="2023-09-01-preview",
    api_key = openai_key
)

def process_static_file(file_path):
    print('Processing static file')
    ext = file_path.split('.')[-1]
    if ext=='xlsx':
        static_df = pd.read_excel(file_path)
    elif ext=='csv':
        static_df = pd.read_csv(file_path)
    else:
        return False, None
    static_df = static_df.dropna(subset=['Product Type']).reset_index(drop=True)
    
    static_df = static_df[['Plant Customer','End Use Description - Customer','Product Name','Product Type','Distinct Machine Types Used','P_LENGTH_MM','P_WIDTH_MM','P_HEIGHT_MM','P_WEIGHT_KG']]
    static_df = static_df.rename(columns={'Product Type': 'Cat','Distinct Machine Types Used':"Machine Type","P_LENGTH_MM":"Length","P_WIDTH_MM":"Width","P_HEIGHT_MM":"Height","P_WEIGHT_KG":"Weight"})
    return True, static_df

status_static, static_df = process_static_file("".replace(' ','%20'))
print('Static File Loaded: ', status_static)

def process_upload_file(file_path):
    print('Processing Upload file')
    ext = file_path.split('.')[-1]
    if ext=='xlsx':
        upload_df = pd.read_excel(file_path.replace(' ',"%20"))
    elif ext=='csv':
        upload_df = pd.read_csv(file_path.replace(' ',"%20"))
    upload_df=upload_df.fillna('')
    return True, upload_df

def generate_batches(df, token_limit):
    current_tokens = 0
    current_batch = []

    for _, row in df.iterrows():
        # Convert row to JSON string
        row_json = f"P: {row['Plant Customer']}. D: {row['End Use Description - Customer']}. PN: {row['Product Name']}. M: {row['Machine Type']}. L: {row['P_LENGTH_MM']}. W: {row['P_WIDTH_MM']}. H: {row['P_HEIGHT_MM']}. W: {row['P_WEIGHT_KG']}"
        row_tokens = num_tokens_from_text(row_json)

        # If adding the current row exceeds the token limit,
        # yield the current batch as a new batch
        # print(current_tokens + row_tokens)
        if current_tokens + row_tokens > token_limit and len(current_batch)>0:
            yield current_batch
            current_batch = []
            current_tokens = 0

        # Add row to current batch
        current_batch.append(row_json)
        current_tokens += row_tokens
        # break

    # Yield the last batch if it's not empty
    if current_batch:
        yield current_batch

def process_batch(batch_number, batch, static_json_string):
    print('BATCH NUMBER: ', batch_number)
    messages = [
        {
            "role": "system",
            "content": f"""
            Please categorize each row of text in the input data in a similar fashion to the examples fed below:
            Examples:
            {static_json_string}

            Use only the categories mentioned in the examples above. Do not make up your own category.
            
            give category which matches most fields Plant Customer, description, Product Name, Machine Type,Height, Width, Length, Weight in example data
            The output should contain the Plant Customer, description, Product Name, Machine Type, Height, Width, Length, Weight, its corresponding category which matches most fields Plant Customer, description, Product Name, Machine Type, Height, Width, Length, Weight in example data, and a confidence score of the categroization between 0 to 1 for each row in the following format.
            PC: <Plant Customer>. DESC: <description>. PName: <Product Name>. MType: <Machine Type>. Height1: <Height>. Width1: <Width>. Length1: <Length>. Weight1: <Weight>. Cat: <category>. CS: <confidence_score>
            above format should be strictly followed each time 
            Do not add up extra information
            """
        },
        {
            "role": "user", 
            "content": f"Categorize the following: \n{batch}"
        }
    ]
    response = openai_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=messages,
        temperature=0,
        timeout=60
    )
    return batch_number, response.choices[0].message.content

def process_categories(upload_file_path):
    print('Categorising file')
    status_upload, upload_df = process_upload_file(upload_file_path)
    if status_static and status_upload:
        static_json_string = '\n'.join([f"P: {item['Plant Customer']}. D: {item['End Use Description - Customer']}. PN: {item['Product Name']}. M: {item['Machine Type']}. C: {item['Cat']}. L: {item['Length']}. W: {item['Width']}. H: {item['Height']}. W: {item['Weight']}" for item in static_df.to_dict(orient='records')])
        token_limit = 50
        batches = generate_batches(upload_df, token_limit)
        print('Generated batches')
        op_text = []
        batches_with_ids = [(batch_idx, batch) for batch_idx, batch in enumerate(batches)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_batch, batch_idx, batch, static_json_string): batch_idx for batch_idx, batch in batches_with_ids}
            for future in concurrent.futures.as_completed(futures):
                batch_number, text = future.result()
                op_text.append((batch_number, text))
        op_text.sort(key=lambda x: x[0])

        print(op_text)
        dict_list = [
            {"Plant Customer":item.split('PC: ')[1].split('. DESC: ')[0],"Description": item.split('DESC: ')[1].split('. PName: ')[0],"Product Name":item.split('PName: ')[1].split('. MType: ')[0],"Machine Type":item.split('MType: ')[1].split('. Height1: ')[0],"Height":item.split('Height1: ')[1].split('. Width1: ')[0],"Width":item.split('Width1: ')[1].split('. Length1: ')[0],"Length":item.split('Length1: ')[1].split('. Weight1: ')[0],"Weight":item.split('Weight1: ')[1].split('. Cat: ')[0], "Category": item.split('Cat: ')[1].split('. CS: ')[0], "Confidence_Score": float(item.split('CS: ')[1])}
            for batch in op_text
            for item in batch[1].split("\n")
            if item
        ]
        return True, dict_list
    return False, "File format not supported"

def generate_categorization_summary(category_dict):
    df = pd.DataFrame(category_dict)
    print(df)
    unique_descriptions = df["Description"].unique()
    unique_categories = df["Category"].unique()
    sample_data = str(df.head(2))
    df = df.rename(columns={'Category':'Product Type'})
    frequency_dict = df['Product Type'].value_counts().to_dict()

    # Get the total count of elements in the DataFrame
    total_count = len(df)

    # Convert frequency values to percentages
    percentage_dict = {key: (value / total_count) * 100 for key, value in frequency_dict.items()}

    # Sort the dictionary based on the highest frequency
    sorted_percentage_dict = dict(sorted(percentage_dict.items(), key=lambda item: item[1], reverse=True))

    messages = [
        {
            "role":"user",
            "content": 
f"""The user has given you a list of box categorization data, that you have previously categorized. Here is some sample data of the categorization:
{sample_data}

The following is a dictionary of the frequencies of the Product Types you have provided:
{sorted_percentage_dict}

Provide an small analysis without numbers in analysis and some numbered action items with Product Type frequency for the user in 150 words.
mention the frequencies in percentages. Keep the summary generalized.
"""
        }
    ]


    openai_key = client.get_secret('OPENAI-API-KEY-SC').value
    openai_api_base = client.get_secret("AZURE-OPENAI-ENDPOINT-SC").value
    DEPLOYMENT_NAME = 'gpt-4-turbo'
    openai_client = openai.AzureOpenAI(
        azure_endpoint = openai_api_base,
        api_version="2023-09-01-preview",
        api_key = openai_key
    )
    response = openai_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=messages,
        temperature=0,
        max_tokens=300,
        timeout=60
    )
    summary = response.choices[0].message.content
    print('SUMMARY: \n', summary)
    return True, summary