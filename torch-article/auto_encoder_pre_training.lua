-- Example of pre-training using an auto-encoder and then
-- loading part of the pre-trained model and fine-tuning for
-- linearly separable classification.

require('torch')
require('nn')
require('optim')
require('gnuplot')
require('lfs')

package.path = package.path .. ";" .. lfs.currentdir() .. '?.lua'
utils = require('utils')
init = require('init')

-- Set up a dataset which will be used for training or testing.
N = 10000
D = 100

batchSize = 10
learningRate = 0.01
momentum = 0.9
weightDecay = 0.05

modelFile = 'model.bin'
if utils.fileExists(modelFile) then -- Load and visualize weights.
  
  -- Classification dataset.
  inputs = torch.Tensor(N, D)
  outputs = torch.Tensor(N, 1)
  line = torch.rand(D)
  
  for i = 1, N do
    inputs[i] = torch.rand(D)
    outputs[i] = inputs[i]:dot(line)
    
    if outputs[i][1] > 0 then
      outputs[i][1] = 1
    else
      outputs[i][1] = 0
    end
  end
  
  trained_model_layers = {1, 3}
  new_model_layers = {1, 3}
  
  -- Define the model.
  model = nn.Sequential()
  model:add(nn.Linear(D, D/5))
  model:add(nn.Tanh())
  model:add(nn.Linear(D/5, D/10))
  model:add(nn.Tanh())
  model:add(nn.Linear(D/10, 1))
  model:add(nn.Sigmoid())
  model = init(model, 'xavier')
  
  -- Change to see the effect of pre-training!
  if true then
    trained_model = torch.load(modelFile)
    utils.copyWeights(trained_model, model, trained_model_layers, new_model_layers)
  end
  
  criterion = nn.BCECriterion()  
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
else -- Train and save a new model.
  
  -- Auto-encoder dataset.
  inputs = torch.Tensor(N, D)
  outputs = torch.Tensor(N, D)

  for i = 1, N do
    outputs[i] = torch.ones(D)
    inputs[i] = torch.cmul(outputs[i], torch.randn(D)*0.05 + 1)
  end

  -- Define the model.
  model = nn.Sequential()
  model:add(nn.Linear(D, D/5))
  model:add(nn.Tanh())
  model:add(nn.Linear(D/5, D/10))
  model:add(nn.Tanh())
  model:add(nn.Linear(D/10, D))
  model = init(model, 'xavier')

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
    print('[Pre-Training] '..t..': '..f[1])
  end
  
  torch.save(modelFile, model)
  print('Training completed; rerun to test.')
end