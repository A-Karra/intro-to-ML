from engine import Value
from NeuralNet import NeuralNet

step = 0.05
num_iterations = 1000

model = NeuralNet(num_in=3, hidden_layers=[4, 4], num_out=3)

input_arr = [[1,5,8],
             [2,4,-6],
             [6,7,-2],
             [-5,2,4],
             [3,10,1]]

output_truth = [[0,1,0],
                [1,0,0],
                [0,0,1],
                [1,0,0],
                [0,1,0]]

output_pred = [model(input) for input in input_arr]
output_pred_data = [[round(float(val.data), 2) for val in row] for row in output_pred]
print(output_pred_data)

for i in range(num_iterations):
    output_pred = [model(input) for input in input_arr]
    
    losses = []
    for pred_row, true_row in zip(output_pred, output_truth):
        for pred, true in zip(pred_row, true_row):
            true_val = Value(true)
            diff = pred - true_val   # needs __sub__
            losses.append(diff * diff)

    loss = sum(losses, Value(0))
    
    loss.zero_grad()
    loss.backward()
    params = model.parameters()
    for p in params:
        p.data -= step * p.grad
    print(loss.data)

output_pred = [model(input) for input in input_arr]
output_pred_data = [[round(float(val.data), 2) for val in row] for row in output_pred]
print(output_pred_data)