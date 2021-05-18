#  ===========================================================================
#  @file:   connect_database.py
#  @brief:  Connecting to SQL database from python saved in azure cloud
#  @author: Johan Mendez
#  @date:   13/05/2021
#  @email:  johan.mendez@databiz.co
#  @status: Debug
#  @detail: version 1.0
#  ===========================================================================
import pyodbc


def connect(server, database, username, password):
	driver = "{ODBC Driver 17 for SQL Server}"
	cnxn   = pyodbc.connect(r"DRIVER="+driver+
							";SERVER="+server+
							";PORT=1433;DATABASE="+database+
							";UID="+username+
							";PWD="+password)
	return cnxn




