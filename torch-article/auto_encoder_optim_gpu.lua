-- Simple auto-encoder example.
-- Uses optim (https://github.com/torch/optim) and GPU for training.

require('math')
require('torch')
require('nn')
require('optim')
require('xlua')

-- For CUDA:
-- - convert the model to CUDA;
-- - convert input and output to CUDA;
-- - convert parameters and gradParameters to CUDA;
-- - convert the criterion to CUDA.
require('cunn')

N = 10000
D = 100

inputs = torch.Tensor(N, D)
outputs = torch.Tensor(N, D)

for i = 1, N do 
  outputs[i] = torch.randn(D)
  inputs[i] = torch.cmul(outputs[i], torch.randn(D)*0.05 + 1)
  
  if i%100 == 0 then
    print('[Data] '..i)
  end
end

model = nn.Sequential()
model:add(nn.Linear(D, 3*D))
model:add(nn.Tanh())
model:add(nn.Linear(3*D, D))

-- !
model = model:cuda()

batch_size = 10
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.05
criterion = nn.AbsCriterion()  
criterion = criterion:cuda()

parameters, gradParameters = model:getParameters()

-- !
parameters = parameters:cuda()
gradParameters = gradParameters:cuda()

T = 2500
for t = 1, T do
  
  -- Sample a random batch from the dataset.
  local shuffle = torch.randperm(N)
  shuffle = shuffle:narrow(1, 1, batch_size)
  shuffle = shuffle:long()
  
  local input = inputs:index(1, shuffle)
  local output = outputs:index(1, shuffle)
  
  -- !
  input = input:cuda()
  output = output:cuda()
  
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
     local pred = model:forward(input) -- pred will be CUDA Tensor 
     local f = criterion:forward(pred, output)

     -- Estimate df/dW.
     local df_do = criterion:backward(pred, output)
     model:backward(input, df_do)

     -- weight decay
     if weight_decay > 0 then
        f = f + weight_decay * torch.norm(parameters,2)^2/2
        gradParameters:add(parameters:clone():mul(weight_decay))
     end

     -- return f and df/dX
     return f, gradParameters
  end
  
  sgd_state = sgd_state or {
      learningRate = learning_rate,
      momentum = momentum,
      learningRateDecay = 5e-7
  }
  
  -- Returns the new parameters and the objective evaluated
  -- before the update.
  p, f = optim.sgd(feval, parameters, sgd_state)
  
  -- xlua.progress(t, T)
  print('[Training] '..t..': '..f[1])
end