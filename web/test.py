import pyodbc

sql_connection = pyodbc.connect(driver="{ODBC Driver 17 for SQL Server}",
                                    host="10.91.13.222\\\\MSSQLSERVER2017,1533", database="TDKT_VinhLong",
                                    user="tdkt", password="123@123axxx")
cursor = sql_connection.cursor()