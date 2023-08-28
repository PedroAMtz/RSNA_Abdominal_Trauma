import sqlite3
import pandas as pd

connection = sqlite3.connect("training_data.db")

sql = pd.read_sql_query("SELECT * FROM base_data", connection)
df = pd.DataFrame(sql, columns =["Patient_id","Series_id", "Patient_paths", "Patient_category"])

def string_to_list(string_repr):
    return eval(string_repr)

df['Patient_paths'] = df['Patient_paths'].apply(string_to_list)