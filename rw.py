import xlrd
import pandas as pd

data = pd.read_excel('energySet/second_energy_set.xlsx')
df = pd.DataFrame(data, columns=['Tagname', 'Timestamp', 'Value'])
#print(df)
#df.dropna(inplace=True)
print(len(df))
new_df = pd.DataFrame({"Tagname":[], "Timestamp":[], "Value":[]})
temp = new_df
sub = 'ELEUV0214_SM31_0214L2_61'
#print(df["Tagname"][4])
for columns in range(len(df)):
    if sub in (df["Tagname"][columns]):
        if df["Value"][columns] != 0:
            df_2 = pd.DataFrame({"Tagname":[df["Tagname"][columns]], "Timestamp":[df["Timestamp"][columns]], "Value":[df["Value"][columns]]})
            temp = temp.append(df_2)
        else:
            continue
    else:
        continue

print(temp)
result = temp.sort_values('Timestamp')
print(result)
result.to_excel('second_set_ELEUV0214_SM31_0214L2_61.xlsx')
#print(df)


