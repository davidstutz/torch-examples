-- Auto-encoder example adapted from examples in the Torch docs using GPU.

require('socket')
require('math')
require('torch')

-- that's the only difference to run on GPU besides converting
-- the model and the input to CUDA
require('cunn')
require('lfs')

package.path = package.path .. ";" .. lfs.currentdir() .. '?.lua'
init = require('init')

--- Get the current milliseconds for benchmarking.
-- http://stackoverflow.com/questions/463101/lua-current-time-in-milliseconds
-- @return milliseconds
function milliseconds()
  return socket.gettime()*1000
end

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

-- for CUDA
-- you will notice that the execution stops here for a bit,
-- an alternative would also be to just put the current batch
-- on the GPU!
--inputs = inputs:cuda()
--outputs = outputs:cuda()

model = nn.Sequential()
model:add(nn.Linear(D, D/2))
model:add(nn.Tanh())
model:add(nn.Linear(D/2, D))
model = init(model, 'xavier')
model = model:cuda()

-- for CUDA
criterion = nn.AbsCriterion() 
criterion = criterion:cuda()

batch_size = 10
learning_rate = 0.01

milliseconds_start = milliseconds()
for i = 1,2500 do
  -- sample a random batch from the dataset
  local shuffle = torch.randperm(N)
  shuffle = shuffle:narrow(1, 1, batch_size)
  shuffle = shuffle:long()
  
  local input = inputs:index(1, shuffle)
  local output = outputs:index(1, shuffle)
  
  -- !
  input = input:cuda()
  output = output:cuda()
  
  -- feed it to the neural network and the criterion
  local loss = criterion:forward(model:forward(input), output)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  model:zeroGradParameters()
  -- (2) accumulate gradients
  model:backward(input, criterion:backward(model.output, output))
  -- (3) update parameters with a 0.01 learning rate
  model:updateParameters(learning_rate)
  
  print('['..i..']: '..loss)
end

milliseconds_end = milliseconds()
print('Took '..(milliseconds_end-milliseconds_start)..' milliseconds.')