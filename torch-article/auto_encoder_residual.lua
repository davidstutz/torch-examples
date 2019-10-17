-- Auto-encoder example usign residual units - residual units do not really make
-- sense in this context, but just as example.
-- Some good examples for residual units:
-- - https://github.com/arunpatala/residual.mnist
-- - https://github.com/gcr/torch-residual-networks

require('math')
require('torch')
require('nn')
require('optim')
require('lfs')

package.path = package.path .. ";" .. lfs.currentdir() .. '?.lua'
init = require('init')

N = 1000
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

--- Simple residual layer.
-- @param input the input model
-- @param nInput input units of linear layer
-- @param nOutput output units of linear layer
-- @return model with residual layer
function addResidualLayer(input, nInput, nOutput)
  local cat = nn.ConcatTable()
  local unit = nn.Linear(nInput, nOutput)
  cat:add(unit)
  cat:add(nn.Identity())

  input:add(cat)
  input:add(nn.CAddTable())
  input:add(nn.Tanh(true))
  return input
end

model = nn.Sequential()
model = addResidualLayer(model, D, D)
model:add(nn.Linear(D, D))
model = init(model, 'xavier')

batchSize = 10
learningRate = 0.01
momentum = 0.9
weightDecay = 0.05
criterion = nn.AbsCriterion()  

parameters, gradParameters = model:getParameters()

for t = 1, 2500 do
  
  -- Sample a random batch from the dataset.
  local shuffle = torch.randperm(N)
  shuffle = shuffle:narrow(1, 1, batchSize)
  shuffle = shuffle:long()
  
  local input = inputs:index(1, shuffle)
  local output = outputs:index(1, shuffle)
  
  --- Definition of the objective on the current mini-batch.
  -- This will be the objective fed to the optimization algorithm.
  -- @param x input parameters
  -- @return object value, gradients
  local feval = function(x)

     -- Get new parameters.
     if x ~= parameters then
        parameters:copy(x)
     end

     -- Reset gradients
     gradParameters:zero()

     -- Evaluate function on mini-batch.
     local pred = model:forward(input)
     local f = criterion:forward(input, output)

     -- Estimate df/dW.
     local df_do = criterion:backward(pred, output)
     model:backward(input, df_do)

     -- weight decay
     if weightDecay > 0 then
        f = f + weightDecay * torch.norm(parameters,2)^2/2
        gradParameters:add(parameters:clone():mul(weightDecay))
     end

     -- return f and df/dX
     return f, gradParameters
  end
  
  sgd_state = sgd_state or {
      learningRate = learningRate,
      momentum = momentum,
      learningRateDecay = 5e-7
  }
  
  -- Returns the new parameters and the objective evaluated
  -- before the update.
  p, f = optim.sgd(feval, parameters, sgd_state)
  
  print('[Training] '..t..': '..f[1])
end