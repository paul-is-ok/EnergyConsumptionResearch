import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# reading in the filtered data set to train and test
energy_data = pd.read_excel('ELEUV0214_SM31_0214L2_61.xlsx')
df = pd.DataFrame(energy_data, columns=["Tagname", "Timestamp", "Value"])
print(df)
# find out how many rows and columns the data set has
print(df.shape)

timestamp = []
value = []
for row in range(len(df)):
    timestamp.append(df["Timestamp"][row])
    value.append(df["Value"][row])

print(timestamp)
print(value)
# creating a visual graph of the data
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.plot(timestamp, value)
plt.title('Energy Consumption vs Time')
plt.ylabel('Energy Consumption')
plt.xlabel('Time')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
#print(plt.plot(df["Value"]))
plt.show()

#print(df.columns)
# preprocessing the data
all_data = df["Value"].values.astype(float)
#print(all_data)

# deciding whhere to determine having the training and testing data set
split = 0.8
test_data_size = int(len(df) * split)
# creating the training and testing data set
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

# print(len(train_data))
# print(len(test_data))

#print(test_data)
# tanh function to control flow of neural networks
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
#print(train_data_normalized)

#print(train_data_normalized[:5])
#print(train_data_normalized[-5:])

# converting to tensor
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

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


train_inout_seq1 = create_inout_sequences(train_data_normalized, train_window)

# print(train_inout_seq[:5])


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
    for seq, labels in train_inout_seq1:
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


# convert back to more readable form
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1,1))
print(actual_predictions)


# mean_absolute_error(y_pred, actual_predictions)
x = np.arange(269, 369, 1)
print(x)

plt.title('Energy Consumption vs Time')
plt.ylabel('Energy Consumption')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(df['Value'])

plt.plot(x, actual_predictions)
plt.show()
