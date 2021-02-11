import pandas as pd

import mysql.connector

cnx = mysql.connector.connect(user='tommy', password='Password*8',
                              host='51.104.203.80',
                              database='bbg_data')

df = pd.read_sql_query("SELECT * FROM bbg_data_daily", cnx)

cnx.close()

xau = df.loc[df["ticker"] == "XAU CURNCY"]
