from pymongo import MongoClient,errors
from pymongo.write_concern import WriteConcern
import logging
from pymongo.write_concern import WriteConcern
import os
from common.keyvault_connection import get_conn

# Get the connection string from the environment variable
client=get_conn()
connection_string = client.get_secret('CUSTOMCONNSTR-CONNECTION-STRING').value


class Database:
    def __init__(self, collection_name,database_name):
        client = MongoClient(connection_string)
        self.db = client[database_name]
        self.collection = self.db[collection_name]
    def insert_single_record(self, record):
        try:
            result = self.collection.insert_one(record)
            if result:
                id = result.inserted_id                
                if id:
                    print('Successfully inserted the record into Database.')
                    return True, id            
                print('Failed to insert the record into Database.')
            return False, 'Failed to insert the record into Database.'
        except errors.DuplicateKeyError:
            return False,'Duplicate Id Error'        
        except Exception as error:
            print(f'Falied Database Methods -> insert_single_record. Error:{error}')

    def delete_single_record(self, id):
        try:
            myquery = {'_id':id}
            result = self.collection.delete_one(myquery)
            if result:
                if result.deleted_count:
                    print('Successfully deleted a record.')
                    return True, None            
                print('Failed to delete the record from Database.')
            return False, 'Failed to delete the record from Database.'        
        except Exception as error:
            print(f'Falied Database Methods -> delete_single_record. Error:{error}')
    
    def delete_record_withquery(self, myquery):
        try:
            result = self.collection.delete_many(myquery)
            if result:
                if result.deleted_count:
                    print('Successfully deleted a record.')
                    return True, None            
                print('Failed to delete the record from Database.')
            return False, 'Failed to delete the record from Database.'        
        except Exception as error:
            print(f'Falied Database Methods -> delete_single_record. Error:{error}')
    
    def delete_all_record(self,query):
        try:
            result = self.collection.with_options(write_concern=WriteConcern(w="majority",j=True)).delete_many(query)
            if result:
                print(f'Successfully deleted {result.deleted_count} record/s from Database.')
                return True, f"Successfully deleted {result.deleted_count} record/s from Database."            
            return False, 'Failed to delete record from Database.'        
        except Exception as error:
            print(f'Falied Database Methods -> delete_all_record. Error:{error}')

    def fetch_one_record(self, filter, cols = {}):
        try:
            result = self.collection.find_one(filter,cols)
            if result is not None and len(result) > 0 :
                print('Successfully fetched a record.')
                return True, result            
            print('Failed to fetch the record from Database.')
            return False, 'Failed to fetch the record from Database.'        
        except Exception as error:
            print(f'Failed Database Methods -> fetch_one_record. Error:{error}')
    def fetch_all_records(self,filter = {},cols = {}):
        try:
            result = list(self.collection.find(filter,
                        cols))
            if len(result) > 0 :
                print('Successfully fetched a record.')
                return True, result            
            print('No records found.')
            return False, 'No records found.'        
        except Exception as error:
            print(f'Falied Database Methods -> fetch_all_records. Error: {error}')
            return False,'Failed to fetch records from database'    
    
    def update_one_record(self,id,records):
        try:
            query = {"_id":id}
            data = {"$set" : records}
            result = self.collection.update_one(query,data)
            if result.matched_count:
                if result.modified_count > 0:
                    print("record updated successfully.")
                    return True, "record updated successfully." 
                print("Record left unaltered.")
                return True, "Record left unaltered."
            print("Failed to update record. Reason: ")           
            return False, "Failed to update record."        
        except Exception as error:
            print(f'Falied Database Methods -> update_one_records. Error:{error}')

    def update_record(self,query,records,condition=None):
        try:
            data = {"$set" : records}
            if condition!=None:
                result = self.collection.update_one(query,{**data, **condition})
            else:
                result = self.collection.update_one(query,data)
            if result.matched_count:
                if result.modified_count > 0:
                    return True, "record updated successfully." 
                return True, "Record left unaltered."           
            return False, "Failed to update record."        
        except Exception as error:
            print(f'Falied Database Methods -> update_record. Error:{error}')
    
    def fetch_aggregate(self,query):
        try:
            result = list(self.collection.aggregate(query))
            if len(result) > 0 :
                print('Successfully fetched a record.')
                return True, result            
            print('No records found.')
            return False, 'No records found.'        
        except Exception as error:
            print(f'Falied Database Methods -> fetch_all_records. Error: {error}')
            return False,'Failed to fetch records from database' 
    
    def increment_one_record(self,id,optional_field = None):
        try:
            query = {"_id":id}
            data = {"$inc" : {"click_count":1}}
            if optional_field is not None:
                data['$inc'][f"category_click.{optional_field}"] = 1
            result = self.collection.update_one(query,data)
            if result.matched_count:
                if result.modified_count > 0:
                    return True, "Click count incremented successfully." 
                return True, "Record left unaltered."           
            return False, "Failed to update record."        
        except Exception as error:
            print(f'Falied Database Methods -> increment_one_record. Error:{error}')
    
    def append_data(self,filter,query):
        try:
            result = self.collection.update_one(filter,query)
            if result.matched_count:
                if result.modified_count > 0:
                    return True, "Record Updated sucessfully." 
                return True, "Record left unaltered."           
            return False, "Failed to update record."        
        except Exception as error:
            print(f'Falied Database Methods -> append_data. Error:{error}')
