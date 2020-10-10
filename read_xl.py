import pandas as pd
import xlrd
import torch
import torch.nn as nn
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
import numpy
import re

# filtering through list based on substring
def filter(string, substring):
    return [str for str in string if
            any(sub in str for sub in substring)]

# reading in data from excel and placing data in list
data = r'newEnergy.xls'
data_read = pd.read_excel(data)
# print(data_read)
tagname_list = list(data_read)
print(tagname_list)
filter120 = ['120']
filter480 = ['480']
filtered_120 = filter(tagname_list, filter120)
#print(filtered_120)
filtered_480 = filter(tagname_list, filter480)
data_120 = []
data_480 = []

#for lines in data_read:
#    if data_read[lines] in filter120:
#        data_120.append(data_read[lines])
#    elif data_read[lines] in filter480:
#        data_480.append(data_read[lines])
#    else:
#        continue

#print(data_120)



# creating a graphical model of the data given for 120
#fig_size = plt.rcParams["figure.figsize"]
#fig_size[0] = 15
#fig_size[1] = 5
#plt.rcParams["figure.figsize"] = fig_size

#plt.title('120 Energy Consumption')
##plt.ylabel('Time of Day')
#plt.xlabel('Device energy used')
#plt.grid = True

all_data = data_read['Value'].values.astype(float)
print(all_data)
# creating training and testing data sets
test_data_size = 30
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

print(len(train_data))
print(len(test_data))


scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

print(train_data_normalized[:5])
print(train_data_normalized[-5:])

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)


# creating testing set
train_window = 30

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

print(train_inout_seq[:5])


#creating LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)