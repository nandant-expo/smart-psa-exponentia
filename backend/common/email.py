from fastapi_mail import FastMail, MessageSchema,ConnectionConfig
import os
from common.keyvault_connection import get_conn

client=get_conn()
async def verify_email(recipients,template_body,template_name,title):
    try:
        conf = ConnectionConfig(
            MAIL_USERNAME=client.get_secret('NOREPLYID').value,
            MAIL_FROM=client.get_secret('NOREPLYID').value,
            MAIL_PASSWORD=client.get_secret('NOREPLYPASS').value,
            MAIL_PORT=client.get_secret('SMTPPORT').value,
            MAIL_SERVER=client.get_secret('SMTPHOST').value,
            MAIL_TLS=True,
            MAIL_SSL=False,
            TEMPLATE_FOLDER='./templates/'
            )
        
        message = MessageSchema(
            subject=title,
            recipients=[recipients],
            template_body=template_body,
            subtype='html',
        )
        
        fm = FastMail(conf)
        
        try:
            await fm.send_message(message,template_name=template_name)
        except Exception as error:
            return False,error

        return True,"Success!"
    except Exception as error:
        return False,error