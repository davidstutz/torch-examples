-- Variational auto-encoder.

require('math')
require('torch')
require('nn')
require('cunn')
require('optim')
require('image')

-- (1) The noise layer adds Bernoulli noise, i.e. flips pixels,
-- randomly with the given probability.
-- Depending on the input, other noise models com einto play, e.g.
-- Gussian noise fo rcontinuous variables.
--- @class SaltPepperNoise
local BernoulliNoise, BernoulliNoiseParent = torch.class('nn.BernoulliNoise', 'nn.Module')

--- Initialize.
-- @param p probability of salt and pepper
function BernoulliNoise:__init(p)
  self.p = p or 0.05
end

--- Compute forward pass, i.e. threshold to 1 at 0.1.
-- @param input layer input
-- @param output
function BernoulliNoise:updateOutput(input)
  -- (1.1) Note that there is a difference between training and testing
  -- if required.
  -- Note, however, that below we do not use this.
  if self.train ~= false and self.p > 0 then
    local rand = torch.rand(input:size())

    if input.__typename == 'torch.CudaTensor' then
      rand = rand:cuda()
    end

    rand[rand:gt(self.p)] = 0
    rand[rand:lt(self.p)] = 1

    self.output = input:clone()
    self.output[rand:eq(1)] = 1 - self.output[rand:eq(1)]
  else
    self.output = input
  end

  return self.output
end

--- Compute the backward pass.
-- @param input original input
-- @param gradOutput gradients of top layer
-- @return gradients with respect to input
function BernoulliNoise:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end

--- @class KullbackLeiblerDivergence
local KullbackLeiblerDivergence, KullbackLeiblerDivergenceParent = torch.class('nn.KullbackLeiblerDivergence', 'nn.Module')

--- Initialize.
-- @param lambda weight of loss
-- @param sizeAverage
function KullbackLeiblerDivergence:__init(lambda, sizeAverage)
  self.lambda = lambda or 1
  self.sizeAverage = sizeAverage or false
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
  self.loss = self.lambda * 0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logVar) - 1 - logVar)

  if self.sizeAverage then
    self.loss = self.loss/(input[1]:size(1)*input[1]:size(2))
  end

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
  self.gradInput[1] = self.lambda*mean
  self.gradInput[2] = self.lambda*0.5*(torch.exp(logVar) - 1)

  if self.sizeAverage then
    self.gradInput[1] = self.gradInput[1]/(input[1]:size(1)*input[1]:size(2))
    self.gradInput[2] = self.gradInput[2]/(input[2]:size(1)*input[2]:size(2))
  end

  self.gradInput[1] = self.gradInput[1] + gradOutput[1]
  self.gradInput[2] = self.gradInput[2] + gradOutput[2]

  return self.gradInput
end

--- @class ReparameterizationSampler
local ReparameterizationSampler, ReparameterizationSamplerParent = torch.class('nn.ReparameterizationSampler', 'nn.Module')

function ReparameterizationSampler:__init()

end

--- Sample from the provided mean and variance using the reparameterization trick.
-- @param input table of two elements, mean and log variance
-- @return sample
function ReparameterizationSampler:updateOutput(input)
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
function ReparameterizationSampler:updateGradInput(input, gradOutput)
  self.gradInput = {}

  local _, logVar = table.unpack(input)
  self.gradInput[1] = gradOutput
  self.gradInput[2] = torch.cmul(torch.cmul(0.5*torch.exp(0.5*logVar), self.eps), gradOutput)

  return self.gradInput
end

-- Data parameters.
H = 24
W = 24
rH = 8
rW = 8
N = 50000

-- Fix random seed.
torch.manualSeed(1)

inputs = torch.Tensor(N, 1, H, W):fill(0)
for i = 1, N do
  local h = torch.random(rH, rH)
  local w = torch.random(rW, rW)
  local aH = torch.random(1, H - h)
  local aW = torch.random(1, W - w)
  inputs[i][1]:sub(aH, aH + h, aW, aW + w):fill(1)
end

outputs = inputs:clone()

-- (2) The encoder just includes the noise layer as very first layer.
hidden = math.floor(2*H*W)
encoder = nn.Sequential()
encoder:add(nn.BernoulliNoise(0.1))
encoder:add(nn.View(1*H*W))
encoder:add(nn.Linear(1*H*W, hidden))
--encoder:add(nn.BatchNormalization(hidden))
encoder:add(nn.ReLU(true))
encoder:add(nn.Linear(hidden, hidden))
--encoder:add(nn.BatchNormalization(hidden))
encoder:add(nn.ReLU(true))

code = 2
encoder:add(nn.View(hidden))
meanLogVar = nn.ConcatTable()
meanLogVar:add(nn.Linear(hidden, code)) -- Mean of the hidden code.
meanLogVar:add(nn.Linear(hidden, code)) -- Variance of the hidden code (diagonal variance matrix).
encoder:add(meanLogVar)

decoder = nn.Sequential()
decoder:add(nn.Linear(code, hidden))
--decoder:add(nn.BatchNormalization(hidden))
decoder:add(nn.ReLU(true))
decoder:add(nn.Linear(hidden, hidden))
--decoder:add(nn.BatchNormalization(hidden))
decoder:add(nn.ReLU(true))
decoder:add(nn.Linear(hidden, 1*H*W))
decoder:add(nn.View(1, H, W))
decoder:add(nn.Sigmoid(true))

model = nn.Sequential()
model:add(encoder)
KLD = nn.KullbackLeiblerDivergence()
model:add(KLD)
model:add(nn.ReparameterizationSampler())
model:add(decoder)
model = model:cuda()

criterion = nn.BCECriterion()
criterion.sizeAverage = false
criterion = criterion:cuda()

parameters, gradParameters = model:getParameters()
parameters = parameters:cuda()
gradParameters = gradParameters:cuda()

batchSize = 16
learningRate = 0.001
epochs = 10
iterations = epochs*math.floor(N/batchSize)
lossIterations = 50 -- in which interval to report training
protocol = torch.Tensor(iterations, 5):fill(0)

for t = 1, iterations do

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

    -- Reset gradients.
    gradParameters:zero()

    -- Evaluate function on mini-batch.
    local pred = model:forward(input)
    local f = criterion:forward(pred, output)

    protocol[t][1] = f
    protocol[t][2] = KLD.loss
    protocol[t][3] = torch.mean(meanLogVar.output[1])
    protocol[t][4] = torch.std(meanLogVar.output[2])
    protocol[t][5] = torch.mean(meanLogVar.output[2])

    -- Estimate df/dW.
    local df_do = criterion:backward(pred, output)
    model:backward(input, df_do)

    -- return f and df/dX
    return f, gradParameters
  end

  adamState = adamState or {
      learningRate = learningRate,
      momentum = 0,
      learningRateDecay = 0.01
  }

  -- Returns the new parameters and the objective evaluated
  -- before the update.
  p, f = optim.adam(feval, parameters, adamState)

  if t%lossIterations == 0 then
    local loss = torch.mean(protocol:narrow(2, 1, 1):narrow(1, t - lossIterations + 1, lossIterations))
    local KLDLoss = torch.mean(protocol:narrow(2, 2, 1):narrow(1, t - lossIterations + 1, lossIterations))
    local mean = torch.mean(protocol:narrow(2, 3, 1):narrow(1, t - lossIterations + 1, lossIterations))
    local std = torch.mean(protocol:narrow(2, 4, 1):narrow(1, t - lossIterations + 1, lossIterations))
    local logvar = torch.mean(protocol:narrow(2, 5, 1):narrow(1, t - lossIterations + 1, lossIterations))
    print('[Training] ' .. t .. '/' .. iterations .. ': ' .. loss .. ' | ' .. KLDLoss .. ' | ' .. mean .. ' | ' .. std .. ' | ' .. logvar)
  end
end

interpolations = torch.Tensor(20 * H, 20 * W)
step = 0.05

-- Sample 20 x 20 points
for i = 1, 20  do
  for j = 1, 20 do
    local sample = torch.Tensor({2 * i * step - 21 * step, 2 * j * step - 21 * step}):view(1, code)
    sample = sample:cuda()
    local interpolation = decoder:forward(sample)
    interpolation = interpolation:float()
    interpolations[{{(i - 1) * H + 1, i * H}, {(j - 1) * W + 1, j * W}}] = interpolation
  end
end

image.save('interpolations.png', interpolations)