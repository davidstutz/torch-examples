-- Convolutional auto encoder example.

require('torch')
require('nn')
require('optim')
require('lfs')

N = 1000
C = 1
H = 8 -- divisible by four
W = 8 -- divisible by four

inputs = torch.Tensor(N, C, H, W)
outputs = torch.Tensor(N, C, H, W)

for i = 1, N do
  -- (1) A binary image with a white square in the middle is
  -- generated and salt and pepper noise is added.
  outputs[i] = torch.Tensor(C, H, W):fill(0)
  outputs[i]:sub(1, 1, H/2 - 2, H/2 + 2, W/2 - 2, H/2 + 2):fill(1)
  inputs[i] = outputs[i]
  
  -- Random indices to set 0.
  zeroIndices = torch.Tensor(C, H, W):uniform():mul(1.05):floor()
  inputs[i][zeroIndices:eq(1)] = 0
  
  oneIndices = torch.Tensor(C, H, W):uniform():mul(1.05):floor()
  inputs[i][oneIndices:eq(1)] = 1
end

-- (1) The convolutional auto-encoder follows the general design in the
-- literature and is analogous to non-convolutional auto-encoders.
-- It consists of an encoder, a bottleneck (the code) and a decoder.
-- (1.1) The encoder applies spatial convolutions and max pooling
-- to reduce the feature map size.
model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(1, 8, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
model:add(nn.SpatialConvolutionMM(8, 8, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

-- (1.2) The last feature map is now resized to a vector
-- using nn.View. The number of units is simultaneously the
-- dimensionality of the code.
hidden = H/4*W/4*8
model:add(nn.View(hidden))
model:add(nn.Linear(hidden, hidden))
model:add(nn.ReLU(true))

-- (1.3) The decode starts by reshaping the linear input
-- and then applies upsampling and convolutions to reconstruct
-- the input.
model:add(nn.View(8, H/4, W/4))
model:add(nn.SpatialUpSamplingNearest(2))
model:add(nn.SpatialConvolutionMM(8, 8, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialUpSamplingNearest(2))
model:add(nn.SpatialConvolutionMM(8, 1, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

batchSize = 10
learningRate = 0.01
momentum = 0.9
weightDecay = 0.05
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

     -- weight decay
     if weightDecay > 0 then
        f = f + weightDecay * torch.norm(parameters,2)^2/2
        gradParameters:add(parameters:clone():mul(weightDecay))
     end

     -- return f and df/dX
     return f, gradParameters
  end
  
  sgdState = sgdState or {
      learningRate = learningRate,
      momentum = momentum,
      learningRateDecay = 5e-7
  }
  
  -- Returns the new parameters and the objective evaluated
  -- before the update.
  p, f = optim.sgd(feval, parameters, sgdState)
  
  print('[Training] '..t..': '..f[1])
end