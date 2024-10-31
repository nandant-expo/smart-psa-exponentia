import logging
import csv
from datetime import datetime
from abc import abstractmethod
from io import StringIO
from common.keyvault_connection import get_conn
from azure.storage.blob import BlobServiceClient
# import atexit
from concurrent.futures import ThreadPoolExecutor
import threading

keyvault=get_conn()

def initiate_logger(logger_name: str):
    logger = None
    logger = AzureBlobStorageLogger(logger_name)
   
    
    # Register cleanup function
    # atexit.register(lambda: logger.executor.shutdown(wait=True))  # Ensures all logging tasks complete
    return logger


class Logger():
    def __init__(self, logger_name) -> None:
        current_date = datetime.now().strftime("%d-%m-%Y")
        self.filename = f'logs/Mankind_logs_{current_date}.csv'
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s")
        self.executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self):
        self.executor.shutdown(wait=True)  # Wait for all background tasks to finish
    
    def _log(self, record):
        """ Override this method in subclasses to implement actual logging """
        pass

    def _submit_log(self, record):
        """ Submit logging task to executor """
        self.executor.submit(self._log, record)

    def debug(self, message, *args, **kwargs):
        record = self._create_log_record(logging.DEBUG, message, *args, **kwargs)
        self.logger.debug(message, *args, **kwargs)
        self._submit_log(record)

    def info(self, message, *args, **kwargs):
        record = self._create_log_record(logging.INFO, message, *args, **kwargs)
        self.logger.info(message, *args, **kwargs)
        self._submit_log(record)
    
    def warning(self, message, *args, **kwargs):
        record = self._create_log_record(logging.WARNING, message, *args, **kwargs)
        self.logger.warning(message, *args, **kwargs)
        self._submit_log(record)
    
    def error(self, message, *args, **kwargs):
        record = self._create_log_record(logging.ERROR, message, *args, **kwargs)
        self.logger.error(message, *args, **kwargs)
        self._submit_log(record)
    
    def critical(self, message, *args, **kwargs):
        record = self._create_log_record(logging.CRITICAL, message, *args, **kwargs)
        self.logger.critical(message, *args, **kwargs)
        self._submit_log(record)

    def _create_log_record(self, level, message, *args, **kwargs):
        """ Create a log record with additional attributes """
        log_record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname='',
            lineno=0,
            msg=message,
            args=args,
            exc_info=None
        )
        log_record.funcName = kwargs.get('funcName', 'unknown')
        log_record.api_hit_time = kwargs.get('api_hit_time', '')
        log_record.workspace = kwargs.get('workspace', '')
        log_record.user_id = kwargs.get('user_id', '')
        log_record.user_name = kwargs.get('user_name', '')
        log_record.api_endpoint = kwargs.get('api_endpoint', '')
        log_record.request_method = kwargs.get('request_method', '')
        log_record.status_code = kwargs.get('status_code', '')
        log_record.response_time = kwargs.get('response_time', '')
        log_record.input = str(kwargs.get('input', ''))
        log_record.response = str(kwargs.get('response', ''))
        log_record.model = kwargs.get('model', '')
        log_record.time_taken = kwargs.get('time_taken', '')
        log_record.query = kwargs.get('query', '')
        return log_record
    
class AzureBlobStorageLogger(Logger):
    def __init__(self, logger_name) -> None:
        super().__init__(logger_name)
        azure_blob_handler = AzureBlobStorageHandler(self.filename)
        azure_blob_handler.setFormatter(self.formatter)
        self.logger.addHandler(azure_blob_handler)

class LoggerHandler(logging.Handler):
    def __init__(self, filename) -> None:
        super().__init__()
        self.filename = filename
    
    @abstractmethod
    def emit(self, record):
        pass

    @abstractmethod
    def _is_empty(self):
        pass


class AzureBlobStorageHandler(LoggerHandler):
    def __init__(self, filename):
        super().__init__(filename)
        self.filename = filename
        self.connection_string = keyvault.get_secret('STORAGE-CONNECTION-STRING').value
        self.container_name = keyvault.get_secret("CONTAINER-NAME").value

        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

        if not self.container_client.exists():
            self.container_client.create_container()

    def emit(self, record):
        threading.Thread(target=self._log, args=(record,)).start()

    def _log(self, record):
        log_entry = {
            'Logger Name': record.name,
            'Workspace Name': getattr(record, 'workspace', ''),
            'UserID': getattr(record, 'user_id', ''),
            'UserName': getattr(record, 'user_name', ''),
            'API Endpoint': getattr(record, 'api_endpoint', ''),
            'API Request Method': getattr(record, 'request_method', ''),
            'Function Name': record.funcName,
            'Api/Function Hit TimeStamp': getattr(record, 'api_hit_time', ''),
            'Log TimeStamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S %p'),
            'logging line number': record.lineno,
            'Level': record.levelname,
            'Level Code': record.levelno,
            'Status Code': getattr(record, 'status_code', ''),
            'Message': self.format(record),
            "input_tokens": getattr(record, 'input', ''),
            "output_tokens": getattr(record, 'response', ''),
            "time_taken": getattr(record, 'time_taken', ''),
            "query": getattr(record, 'query', ''),
        }

        output = StringIO()
        csv_writer = csv.DictWriter(output, fieldnames=log_entry.keys())
        
        blob = self.container_client.get_blob_client(self.filename)
        try:
            existing_content = blob.download_blob().readall().decode('utf-8')
            output.write(existing_content)
        except Exception:
            csv_writer.writeheader()

        csv_writer.writerow(log_entry)

        content = output.getvalue()
        blob.upload_blob(content, overwrite=True)