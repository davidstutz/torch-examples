-- Example of saving and loading models.

require('torch')
require('nn')
require('optim')
require('lfs')

package.path = package.path .. ";" .. lfs.currentdir() .. '?.lua'
utils = require('utils')
init = require('init')

-- Set up a dataset which will be used for training or testing.
N = 1000
D = 10

inputs = torch.Tensor(N, D)
outputs = torch.Tensor(N, D)

for i = 1, N do 
  outputs[i] = torch.ones(D)
  inputs[i] = torch.cmul(outputs[i], torch.randn(D)*0.05 + 1)
end

model = nil
modelFile = 'model.bin'
if utils.fileExists(modelFile) then -- Load and test model.
  model = torch.load(modelFile)
  
  -- Test the model.
  preds = model:forward(inputs)
  meanError = torch.mean(outputs - preds)/D
  print('Error:'..meanError)
else -- Train and save a new model.
  
  -- Define the model.
  model = nn.Sequential()
  model:add(nn.Linear(D, D/2))
  model:add(nn.Tanh())
  model:add(nn.Linear(D/2, D))
  model = init(model, 'xavier')

  -- Learning hyperparameters.
  batchSize = 10
  learningRate = 0.01
  momentum = 0.9
  weightDecay = 0.05
  criterion = nn.AbsCriterion()  

  parameters, gradParameters = model:getParameters()

  T = 500
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
  end
  
  torch.save(modelFile, model)
  print('Training completed; rerun to test.')
end