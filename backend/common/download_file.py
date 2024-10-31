import os
import aspose.slides as slides
from datetime import datetime
from io import BytesIO
from common.database import Database
from collections import defaultdict
from azure.storage.blob import BlobServiceClient
from cachetools import TTLCache, cached
from threading import Lock
import requests
import concurrent.futures
from common.keyvault_connection import get_conn

client=get_conn()

storage_account = client.get_secret("STORAGE-ACCOUNT").value
storage_connection_string = client.get_secret("STORAGE-CONNECTION-STRING").value
blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

def upload_merged_file(filename, merged_ppt):
    blob_client = blob_service_client.get_blob_client("slide-data", os.path.join("ppt_downloads", filename))
    blob_client.upload_blob(merged_ppt.getvalue(),overwrite=True)
    blob_url = os.path.join(storage_account, 'slide-data', os.path.join("ppt_downloads", filename))
    print(f"The URL of the blob is: {blob_url}")
    return blob_url

def group_ppts(data,workspace):
    db = Database('pptCollection',workspace)
    output = defaultdict(lambda: {"slide_number": [], "position": []})
    for i, item in enumerate(data):
        output[item["_id"]]["slide_number"].append(item["slide_number"])
        output[item["_id"]]["position"].append(i)
        if "blob_path" not in output[item["_id"]].keys():
            status,url = db.fetch_one_record({"_id":item["_id"]},{'_id':0,'blob_path':1})
            if status:
                output[item["_id"]].update(url)
    return output

def download_blob(blob_url):
    print(f"Downloading {blob_url}...")
    blob_name = blob_url.split("/")[-1]
    response = requests.get(blob_url)
    content = response.content
    # print(blob_name)
    return slides.Presentation(BytesIO(content))

def download_and_read(files):
    file = download_blob(files[1]['blob_path'])
    return (files[0],file)

def merge_slides(slides_group,pres1,filename):
    target_presentation = slides.Presentation()
    for slide in slides_group:
        target_presentation.slides.add_clone(slide)
    slide_size = pres1.slide_size.size
    target_presentation.slide_size.set_size(slide_size.width,slide_size.height,slides.SlideSizeScaleType.DO_NOT_SCALE)
    target_presentation.slides.remove_at(0)
    ppt_bytes = BytesIO()
    filename = f"{filename}.pptx"
    target_presentation.save(ppt_bytes, slides.export.SaveFormat.PPTX)
    url = upload_merged_file(filename,ppt_bytes)
    print("Succesfully Merged and Uploaded !!!")
    return url


def slides_merge(slides_group,single,workspace):
    try:
        filename = f"Custom_download_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if single:
            filename = slides_group[0]['slide_title']
        slides_group = group_ppts(slides_group,workspace)
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            results = [executor.submit(download_and_read,(key,value)) for key,value in slides_group.items()]
            concurrent.futures.wait(results)
        slides_to_add = [result.result()[1].slides[slide_no-1] for result in results for slide_no in slides_group[result.result()[0]]['slide_number']]
        ppt_url = merge_slides(slides_to_add, results[0].result()[1],filename)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: File Uploaded Successfully.")
        return True,ppt_url 
    except Exception as error:
        print(f"Failed:ppt_merge. Error: {error}")
        return False,"Failed to merge ppts."
        