import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array


# reading through the filtered set of data to train and test
energy_data = pd.read_excel('ELEUV0214_SM2_0214HMBA_21.xlsx')
df = pd.DataFrame(energy_data, columns=['Tagname', 'Timestamp', 'Value'])
# print(df)
# print(df.shape)

#timestamp = []
#value = []
#for row in range(len(df)):
#    timestamp.append(df["Timestamp"][row])
#    value.append(df["Value"][row])

#print(timestamp)
#print(value)

x_train, x_test, Y_train, Y_test = train_test_split(df["Timestamp"], df["Value"], test_size=0.2, random_state=0)

# creating a visual graph of the data
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.plot(x_train, Y_train)
plt.title('Energy Consumption vs Time')
plt.ylabel('Energy Consumption')
plt.xlabel('Time')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
#print(plt.plot(df["Value"]))
plt.show()


# preprocessing the data
all_data = df["Value"].values.astype(float)
#print(all_data)


scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(np.array(x_train).reshape(-1, 1))
print(train_data_normalized)
#print(len(train_data_normalized))

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
print(train_data_normalized)


train_window = 10

# return a list of tuples to use to train the data
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

"""
***** This part was where I would try to shuffle the dataset to create a couple different minibatches ******
train_sets = []
train_inout_seq1 = create_inout_sequences(train_data_normalized[0, 109], train_window)
train_inout_seq2 = create_inout_sequences(train_data_normalized[110, 219], train_window)
train_inout_seq3 = create_inout_sequences(train_data_normalized[220, 329], train_window)
train_inout_seq4 = create_inout_sequences(train_data_normalized[330, 439], train_window)
train_sets.append(train_inout_seq1)
train_sets.append(train_inout_seq2)
train_sets.append(train_inout_seq3)
train_sets.append(train_inout_seq4)
"""

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
# creating the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=200, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# training the model
epochs = 200

for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1,1,model.hidden_layer_size),
                                 torch.zeros(1,1,model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


# making predictions with the model
fut_pred = 100

test_inputs = train_data_normalized[-train_window:].tolist()
print(test_inputs)


model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1,1, model.hidden_layer_size),
                        torch.zeros(1,1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())


print(test_inputs[fut_pred:])

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# convert back to more readable form
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1,1))
print(actual_predictions)

#print(mean_absolute_percentage_error(test_inputs, actual_predictions))
