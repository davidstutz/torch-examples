-- Auto-encoder example adapted from the Torch docs.

require('math')
require('torch')
require('nn')

N = 1000
D = 100

-- (1) Setup the dataset; note that the dataset is not bound to
-- the structure used for nn.StochasticTraining anymore.
inputs = torch.Tensor(N, D)
outputs = torch.Tensor(N, D)

for i = 1, N do 
  outputs[i] = torch.ones(D)
  inputs[i] = torch.cmul(outputs[i], torch.randn(D)*0.05 + 1)
  
  if i%100 == 0 then
    print('[Data] '..i)
  end
end

model = nn.Sequential()
model:add(nn.Linear(D, D/2))
model:add(nn.Tanh())
model:add(nn.Linear(D/2, D))

-- (2) Setup the batch size, learning rate and the
-- absolute difference criterion.
batchSize = 10
learningRate = 0.01
criterion = nn.AbsCriterion()  

-- (3) The main loop defining the number of iterations.
for i = 1, 500 do
  -- (3.1) Input and output batch is chosen randomly from the dataset.
  -- This is done by shuffling the indices and selecting a fixed subset.
  local shuffle = torch.randperm(N)
  shuffle = shuffle:narrow(1, 1, batchSize)
  shuffle = shuffle:long()
  
  local input = inputs:index(1, shuffle)
  local output = outputs:index(1, shuffle)
  
  -- (3.2) Forward pass of the network and the criterion.
  local loss = criterion:forward(model:forward(input), output)

  -- (3.3) Zero the accumulated gradients.
  model:zeroGradParameters()
  -- (3.4) Compute gradients for this iteration using a backward pass
  -- of the criterion and the model.
  model:backward(input, criterion:backward(model.output, output))
  -- (3.5) Update the parameters using a gradient descent step.
  model:updateParameters(learningRate)
  
  print('[Training] '..i..': '..loss)
end