import xlrd
import pandas as pd

data = pd.read_csv('energyData-10-12.txt', skiprows=50000, nrows=10000)
data.to_excel('october_sixth_batch.xlsx')
#df = pd.DataFrame(data, columns=['Tagname', 'Timestamp', 'Value'])
#print(df)
#df.dropna(inplace=True)
#print(len(df))
#new_df = pd.DataFrame({"Tagname":[], "Timestamp":[], "Value":[]})
#temp = new_df
#sub = 'ELEUV0214_SM22_0214L2_30'
#print(df["Tagname"][4])
#for columns in range(5000):
#    if sub in (df["Tagname"][columns]):
#        if df["Value"][columns] != 0:
#            df_2 = pd.DataFrame({"Tagname":[df["Tagname"][columns]], "Timestamp":[df["Timestamp"][columns]], "Value":[df["Value"][columns]]})
#            temp = temp.append(df_2)
#       else:
#           continue
#    else:
#        continue

#print(temp)
#result = temp.sort_values('Timestamp')
#print(result)
#result.to_excel('ELEUV0214_SM22_0214L2_30.xlsx')