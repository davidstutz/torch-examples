-- Simple auto-encoder example.
-- Uses optim (https://github.com/torch/optim) for training.

require('math')
require('torch')
require('nn')
require('optim')

N = 10000
D = 100

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

-- (1) In addition to batch size and learning rate, we additionally
-- define a momentum parameter and the weight decay weight.
batchSize = 10
learningRate = 0.01
momentum = 0.9
weightDecay = 0.05
criterion = nn.AbsCriterion()  

-- (2) Get the parameters and its gradients on which to perform stochastic 
-- gradient descent. Note that there are some caveats with getParameters:
-- https://github.com/torch/DEPRECEATED-torch7-distro/issues/33
parameters, gradParameters = model:getParameters()

T = 2500
for t = 1, T do
  
  -- Sample a random batch from the dataset.
  local shuffle = torch.randperm(N)
  shuffle = shuffle:narrow(1, 1, batchSize)
  shuffle = shuffle:long()
  
  local input = inputs:index(1, shuffle)
  local output = outputs:index(1, shuffle)
  
  -- (3) Define the objective function, called feval.
  --- Definition of the objective on the current mini-batch.
  -- This will be the objective fed to the optimization algorithm.
  -- @param x input parameters
  -- @return object value, gradients
  local feval = function(x)
    
    -- Get new parameters.
    if x ~= parameters then
      parameters:copy(x)
    end

    -- (3.1) As before, reset the accumulated gradients.
    gradParameters:zero()

    -- (3.2) Compute the forward pass of the network and the criterion.
    local pred = model:forward(input)
    local f = criterion:forward(pred, output)

    -- (3.3) Estimate the gradients through a backward pass of the
    -- network and criterion.
    local df_do = criterion:backward(pred, output)
    model:backward(input, df_do)

    -- (3.4) Add weight decay if requested.
    if weightDecay > 0 then
      f = f + weightDecay * torch.norm(parameters,2)^2/2
      gradParameters:add(parameters:clone():mul(weightDecay))
    end

    return f, gradParameters
  end
  
  sgdState = sgdState or {
      learningRate = learningRate,
      momentum = momentum
  }
  
  -- (4) Run optim.sgd for one step on the defined objective.
  -- The parameters and the objective value is returned.
  p, f = optim.sgd(feval, parameters, sgdState)
  
  print('[Training] '..t..': '..f[1])
end