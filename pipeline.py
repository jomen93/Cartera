#  ===========================================================================
#  @file: pipeline.py
#  @brief: Main program
#  @author: Johan Mendez
#  @date:   13/05/2021
#  @email:  johan.mendez@databiz.co
#  @status: Debug
#  @detail: version 1.0
#  ===========================================================================


from connect_database import connect
import pandas as pd


server   = "carterasvr.database.windows.net"
database = "cartera"
username = "consulta"
password = "D4t4b1z2.123"

cnxn = connect(server, database, username, password)

payment_query = "SELECT * FROM FAC_PAGO"
payment_data  = pd.read_sql(payment_query, cnxn)

print(payment_data)