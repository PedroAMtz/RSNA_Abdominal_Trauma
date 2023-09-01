import pandas as pd
import sqlite3

def string_to_list(string_repr):
    return eval(string_repr) 

connection = sqlite3.connect("C:/Users/Daniel/Desktop/RSNA_Abdominal_Trauma/local_database/training_data.db")
    # ATTENTION ABOUT THE TABLE FROM THE DB YOU CONNECT!!
sql = pd.read_sql_query("SELECT * FROM base_data", connection)
data = pd.DataFrame(sql, columns =["Patient_id", "Series_id", "Patient_paths", "Patient_category"])    
data['Patient_paths'] = data['Patient_paths'].apply(string_to_list)

data_positive = data[data["Patient_category"] == 1].sample(frac=1, random_state=42)
data_positive = data_positive.reset_index(drop=True)   
data_positive = data_positive.drop_duplicates(subset=['Patient_id'], keep='first')

data_negative = data[data["Patient_category"] == 0].sample(frac=1, random_state=42)
data_negative = data_negative.reset_index(drop=True)
data_negative.drop_duplicates(subset=['Patient_id'], keep='first')
data_negative = data_negative.sample(n=855)

merged_data = pd.concat([data_positive, data_negative], ignore_index=True).sample(frac=1, random_state=42)
merged_data.drop('Patient_paths', axis=1, inplace=True)

merged_data.to_sql(name="balanced_data", con=connection, if_exists="append", index=False)