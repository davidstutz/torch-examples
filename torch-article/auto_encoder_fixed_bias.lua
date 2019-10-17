-- Simple auto-encoder example with fixed bias of linear layers.

require('math')
require('torch')
require('nn')
require('optim')
require('lfs')

local LinearFixedBias, LinearFixedParent = torch.class('nn.LinearFixedBias', 'nn.Linear')

--- Required constructor.
-- @param inputSize number of input units
-- @param outputSize number of output units
-- @param bias whether to use a bias
function LinearFixedBias:__init(inputSize, outputSize, bias)
  LinearFixedParent.__init(self, inputSize, outputSize, bias)
end

--- Avoids accumulating gradient in order to fix the parameters
-- for fine-tuning.
function LinearFixedBias:accGradParameters(input, gradOutput, scale)
  LinearFixedParent.accGradParameters(self, input, gradOutput, scale)
  self.gradBias:fill(0)
end

package.path = package.path .. ";" .. lfs.currentdir() .. '?.lua'
init = require('init')

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
model:add(nn.LinearFixedBias(D, D/2))
model:add(nn.Tanh())
model:add(nn.Linear(D/2, D))

-- As example, set the bias of layer 1 to 0.5 fixed.
model = init(model, 'xavier')
model.modules[1].bias:fill(0.5)

batchSize = 10
learningRate = 0.01
momentum = 0.9
criterion = nn.AbsCriterion()  

parameters, gradParameters = model:getParameters()

T = 2500
for t = 1, T do
  
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
     local f = criterion:forward(pred, output)

     -- Estimate df/dW.
     local df_do = criterion:backward(pred, output)
     model:backward(input, df_do)

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
  
  -- xlua.progress(t, T)
  print('[Training] '..t..': '..f[1])
  print('[Training] '..t..': bias '..torch.norm(model.modules[1].bias, 1)..'/'..torch.norm(model.modules[3].bias, 1))
  print('[Training] '..t..': bias grad '..torch.norm(model.modules[1].gradBias, 1)..'/'..torch.norm(model.modules[3].gradBias, 1))
end