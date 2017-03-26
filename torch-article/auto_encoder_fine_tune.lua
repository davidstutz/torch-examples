-- Example of fine-tuning where weights obtained forman
-- auto-encoder are fixed, and an additional fully connected
-- layer is then trained for classification.

require('torch')
require('nn')
require('optim')

-- (1) LinearFT extends nn.Linear to overwrite accGradParameters
-- in order to fix both weights and biases.
--- @class LinearFT
local LinearFT, LinearFTParent = torch.class('nn.LinearFT', 'nn.Linear')

--- Required constructor.
-- @param inputSize number of input units
-- @param outputSize number of output units
-- @param bias whether to use a bias
function LinearFT:__init(inputSize, outputSize, bias)
  LinearFTParent.__init(self, inputSize, outputSize, bias)
end

--- Avoids accumulating gradient in order to fix the parameters
-- for fine-tuning.
function LinearFT:accGradParameters(input, gradOutput, scale)
  -- Nothing!
end

-- (2) A simple, unsafe method for copying weights. 
-- This version does not check the module types.
--- "Unsafe": Copies the weights of the given layers between two models; does not check
-- that the layers are of the same type; assumes the layers to have .weight and .bias defined.
-- @param modelFrom mode to copy weights from
-- @param modelTo model to copy weights to
-- @param layersFrom layer indices in model_from
-- @param layersTo layer indices in model_to
function copyWeights(modelFrom, modelTo, layersFrom, layersTo)
  assert(#layersFrom == #layersTo)
  
  for i = 1, #layersFrom do
    modelTo.modules[layersTo[i]].weight = modelFrom.modules[layersFrom[i]].weight
    modelTo.modules[layersTo[i]].bias = modelFrom.modules[layersFrom[i]].bias
  end
end

-- Set up a dataset which will be used for training or testing.
N = 10000
D = 100

batchSize = 10
learningRate = 0.01
momentum = 0.9
weightDecay = 0.05

-- (3) The example should be run twice; first, an auto-encoder
-- is trained, then this model is fien tuned for classification.
modelFile = 'model.dat'
if utils.fileExists(modelFile) then
  
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
  
  trainedModelLayers = {1, 3}
  modelLayers = {1, 3}
  
  -- (4) The model for classification fixes the second linear layer of the
  -- auto-encoder using LinearFT.
  -- Define the model.
  model = nn.Sequential()
  model:add(nn.Linear(D, D/5)) -- check that weights change due to LinearFT!
  model:add(nn.Tanh())
  model:add(nn.LinearFT(D/5, D/10))
  model:add(nn.Tanh())
  model:add(nn.Linear(D/10, 1))
  model:add(nn.Sigmoid())
  model = init(model, 'xavier')
  
  -- (5) The auto-encoder is loaded and the weights of the first and
  -- second linear layers are copied.
  -- The remaining part for training is similar to the other
  -- examples.
  trainedModel = torch.load(modelFile)
  -- Copy weights without checking the layer types!
  copyWeights(trainedModel, model, trainedModelLayers, modelLayers)
  
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
    
    state = state or {
      learningRate = learningRate,
      momentum = momentum,
      learningRateDecay = 5e-7
    }
    
    -- Returns the new parameters and the objective evaluated
    -- before the update.
    p, f = optim.sgd(feval, parameters, state)
    print('[Training] ' .. t .. ': ' .. f[1])
    
    -- To check that first layer weights change if using LinearFT as second
    -- layer!
    --print('[Training] '..t..': '..torch.mean(model.modules[1].weight))
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
    
    state = state or {
      learningRate = learningRate,
      momentum = momentum,
      learningRateDecay = 5e-7
    }
    
    -- Returns the new parameters and the objective evaluated
    -- before the update.
    p, f = optim.sgd(feval, parameters, state)
    print('[Pre-Training] ' .. t .. ': ' .. f[1])
  end
  
  torch.save(modelFile, model)
  print('Training completed; rerun to test.')
end