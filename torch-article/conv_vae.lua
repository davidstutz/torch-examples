-- Convolutional auto encoder example.

require('torch')
require('nn')
require('cunn')
require('optim')
require('lfs')
require('image')

--- @class KullbackLeiblerDivergence
local KullbackLeiblerDivergence, KullbackLeiblerDivergenceParent = torch.class('nn.KullbackLeiblerDivergence', 'nn.Module')

--- Initialize.
function KullbackLeiblerDivergence:__init()
  self.loss = nil
end

--- Compute the Kullback-Leiber divergence; however, the input remains
-- unchanged - the divergence is saved in KullBackLeiblerDivergence.loss.
-- @param input table of two elements, mean and log variance
-- @param table of wo elements, mean and log variance
function KullbackLeiblerDivergence:updateOutput(input)
  assert(#input == 2)

  -- Save the loss for monitoring.
  local mean, logVar = table.unpack(input)
  self.loss = 0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logVar) - 1 - logVar)

  self.output = input
  return self.output
end

--- Compute the backward pass of the Kullback-Leibler Divergence.
-- @param input original inpur as table of two elements, mean and log variance
-- @param gradOutput gradients from top layer, table of two elements, mean and log variance
-- @param gradients with respect to input, table of two elements
function KullbackLeiblerDivergence:updateGradInput(input, gradOutput)
  assert(#gradOutput == 2)

  local mean, logVar = table.unpack(input)
  self.gradInput = {}
  self.gradInput[1] = mean + gradOutput[1]
  self.gradInput[2] = 0.5 * (torch.exp(logVar) - 1) + gradOutput[2]

  return self.gradInput
end

--- Sampler
local Sampler, SamplerParent = torch.class('nn.Sampler', 'nn.Module')

--- Initialize.
function Sampler:__init()
  SamplerParent.__init(self)
end

--- Sample from the provided mean and variance using the reparameterization trick.
-- @param input table of two elements, mean and log variance
-- @return sample
function Sampler:updateOutput(input)
  assert(#input == 2)

  local mean, logVar = table.unpack(input)
  self.eps = torch.randn(input[1]:size()):cuda()
  self.output = torch.cmul(torch.exp(0.5*logVar), self.eps) + mean

  return self.output
end

--- Backward pass of the sampler.
-- @param input table of two elements, mean and log variance
-- @param gradOutput gradients of top layer
-- @return gradients with respect to input, table of two elements
function Sampler:updateGradInput(input, gradOutput)
  self.gradInput = {}

  local mean, logVar = table.unpack(input)
  self.gradInput[1] = gradOutput
  self.gradInput[2] = torch.cmul(torch.cmul(0.5*torch.exp(0.5*logVar), self.eps), gradOutput)

  return self.gradInput
end

package.path = package.path .. ";" .. lfs.currentdir() .. '?.lua'
init = require('init')

-- Data parameters.
H = 32
W = 32
rH = 12
rW = 12
N = 100000

-- Fix random seed.
--torch.manualSeed(1)

-- Generate rectangle data.
inputs = torch.Tensor(N, 1, H, W):fill(0)
for i = 1, N do
  local h = torch.random(2, rH)
  local w = torch.random(2, rW)
  local aH = torch.random(1, H - h)
  local aW = torch.random(1, W - w)
  inputs[i][1]:sub(aH, aH + h, aW, aW + w):fill(1)
end

outputs = inputs:clone()

-- Encoder
encoder = nn.Sequential()
encoder:add(nn.SpatialConvolutionMM(1, 4, 3, 3, 1, 1, 1, 1))
encoder:add(nn.SpatialBatchNormalization(4))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))
encoder:add(nn.SpatialConvolutionMM(4, 8, 3, 3, 1, 1, 1, 1))
encoder:add(nn.SpatialBatchNormalization(8))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))
encoder:add(nn.SpatialConvolutionMM(8, 16, 3, 3, 1, 1, 1, 1))
encoder:add(nn.SpatialBatchNormalization(16))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))

hidden = H/8*W/8*16
code = 2
encoder:add(nn.View(hidden))
meanLogVar = nn.ConcatTable()
meanLogVar:add(nn.Linear(hidden, code)) -- Mean of the hidden code.
meanLogVar:add(nn.Linear(hidden, code)) -- Variance of the hidden code (diagonal variance matrix).
encoder:add(meanLogVar)

-- Decoder.
decoder = nn.Sequential()
decoder:add(nn.Linear(code, hidden))
--decoder:add(nn.BatchNormalization(hidden))
decoder:add(nn.ReLU(true))

decoder:add(nn.View(16, H/8, W/8))
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(nn.SpatialConvolutionMM(16, 8, 3, 3, 1, 1, 1, 1))
decoder:add(nn.ReLU(true))
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(nn.SpatialConvolutionMM(8, 4, 3, 3, 1, 1, 1, 1))
decoder:add(nn.ReLU(true))
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(nn.SpatialConvolutionMM(4, 1, 3, 3, 1, 1, 1, 1))
decoder:add(nn.Sigmoid(true))

model = nn.Sequential()
model:add(encoder)

KLD = nn.KullbackLeiblerDivergence()
model:add(KLD)
model:add(nn.Sampler)

model:add(decoder)
model = init(model, 'xavier')
model = model:cuda()

batchSize = 32
learningRate = 0.005
momentum = 0.0
weightDecay = 0.0

criterion = nn.BCECriterion()
-- As the Kullback-Leibler loss is not normalized!
criterion.sizeAverage = false -- !
criterion = criterion:cuda()

parameters, gradParameters = model:getParameters()
parameters = parameters:cuda()
gradParameters = gradParameters:cuda()

T = math.floor(2.5*100000/batchSize)
for t = 1, T do

  -- Sample a random batch from the dataset.
  local shuffle = torch.randperm(N)
  shuffle = shuffle:narrow(1, 1, batchSize)
  shuffle = shuffle:long()

  local input = inputs:index(1, shuffle)
  local output = outputs:index(1, shuffle)

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
    local pred = model:forward(input)
    local f = criterion:forward(pred, output)

    --local abs = nn.AbsCriterion()
    --abs = abs:cuda()
    --print(abs:forward(pred, output))

    -- Estimate df/dW.
    local df_do = criterion:backward(pred, output)
    model:backward(input, df_do)

    -- weight decay
    if weightDecay > 0 then
      f = f + weightDecay * torch.norm(parameters,2)^2/2
      gradParameters:add(parameters:clone():mul(weightDecay))
    end

    -- Add the Kullback-Leibler divergence:
    f = f + KLD.loss

    -- return f and df/dX
    return f, gradParameters
  end

  adamState = adamState or {
      learningRate = learningRate,
      momentum = momentum,
      learningRateDecay = 5e-7
  }

  -- Returns the new parameters and the objective evaluated
  -- before the update.
  p, f = optim.adam(feval, parameters, adamState)

  print('[Training] '..t..': '..f[1])
end

-- Check distribution.
inputs = inputs:cuda()
outputs = encoder:forward(inputs)
outputs[1] = outputs[1]:cuda()
outputs[2] = outputs[2]:float()
print(torch.mean(outputs[1]), torch.std(outputs[1]), torch.mean(outputs[2]))

-- Plot interpolations
interpolations = torch.Tensor(15 * H, 15 * W)
step = 0.05

-- Sample 15 x 15 points
for i = 1, 15  do
  for j = 1, 15 do
    local sample = torch.Tensor({2 * i * step - 16 * step, 2 * j * step - 16 * step}):view(1, code) -- Minibatch of 1 for batch normalisation
    sample = sample:cuda()
    local interpolation = decoder:forward(sample)
    interpolation = interpolation:float()
    interpolations[{{(i-1) * H + 1, i * H}, {(j-1) * W + 1, j * W}}] = interpolation
  end
end

image.save('interpolations.png', interpolations)