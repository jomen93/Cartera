# *****************************************************************************
#  @file connect_database.py
#  
#  @date:   13/05/2021
#  @author: Johan Mendez
#  @email:  johan.mendez@dtabiz.co
#  @status: Debug
# 
#  @brief
#  
# 
# 
#  @detail
# 
#
# 
#  *****************************************************************************

import pyodbc


def connect(server, database, username, password):
	driver = "{ODBC Driver 17 for SQL Server}"
	cnxn   = pyodbc.connect(r"DRIVER="+driver+
							";SERVER="+server+
							";PORT=1433;DATABASE="+database+
							";UID="+username+
							";PWD="+password)
	return cnxn




