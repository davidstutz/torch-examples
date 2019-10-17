-- Denoising variational auto-encoder.

require('math')
require('torch')
require('nn')
require('cunn')
require('optim')
require('image')

-- (1) The Kullback Leibler loss is defined as additional nn module, i.e. layer.
-- In the forward pass, the loss is computed, but the input is passed forward
-- without change.
-- On the backward pass, an additive loss corresponding to the
-- derivative of the Cullback Leibler loss is added to the gradients.
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

  -- (1.1) In the forward pass, mean and log-variance are assumed to be passed as table.
  -- Then the loss is computed as outlined below.
  -- Optionally, the loss is averaged by size.
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

  -- (1.2) In the backward pass, gradients for mean and log-variance are
  -- computed separately.
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

-- (2) The sampler samples a random variable given the mean and standard deviation
-- vector; the samples value will be the input to the decoder.
-- For sampling the reparameterization trick is used which
-- also allows to implement the backward pass.
--- @class ReparameterizationSampler
local ReparameterizationSampler, ReparameterizationSamplerParent = torch.class('nn.ReparameterizationSampler', 'nn.Module')

function ReparameterizationSampler:__init()

end

--- Sample from the provided mean and variance using the reparameterization trick.
-- @param input table of two elements, mean and log variance
-- @return sample
function ReparameterizationSampler:updateOutput(input)
  assert(#input == 2)

  -- (2.1) Forward pass.
  -- Note that the samples assumes CUDA training;
  -- otherwise the lines below might need to be adapted.
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

  -- (2.2) Backward pass.
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

-- (3) The example data will be rectangles of random size which
-- are to be auto-encoded by the VAE.
-- Generate rectangle data.
inputs = torch.Tensor(N, 1, H, W):fill(0)
for i = 1, N do
  local h = torch.random(rH, rH)
  local w = torch.random(rW, rW)
  local aH = torch.random(1, H - h)
  local aW = torch.random(1, W - w)
  inputs[i][1]:sub(aH, aH + h, aW, aW + w):fill(1)
end

outputs = inputs:clone()
print('[Training] created training set')

-- (4) The encoder consists of several linear layers followed by
-- the Kullback Leibler loss, the samples and the docoder; the decoder
-- mirrors the encoder.
-- (4.1) The encoder:
hidden = math.floor(2*H*W)
encoder = nn.Sequential()
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

-- (4.2) The decoder:
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

-- (4.3) The full model, i.e encoder followed by the Kullback Leibler
-- divergence and the reparameterization trick sampler.
model = nn.Sequential()
model:add(encoder)
KLD = nn.KullbackLeiblerDivergence()
model:add(KLD)
model:add(nn.ReparameterizationSampler())
model:add(decoder)
model = model:cuda()
print(model)

-- (4.4) As criterion, a binary cross entropy criterion is used (as
-- for classification), note that this is also discussed in the paper.
-- Note that averaging is turned off in order to automatically weight
-- BCE loss and Kullback-Leibler divergence.
criterion = nn.BCECriterion()
criterion.sizeAverage = false
criterion = criterion:cuda()

parameters, gradParameters = model:getParameters()
parameters = parameters:cuda()
gradParameters = gradParameters:cuda()

-- (5) Training proceeds as for regular networks.
-- The BCE loss and the Kullback Leibler loss are monitored
-- separately.
batchSize = 16
learningRate = 0.001
epochs = 10
iterations = epochs*math.floor(N/batchSize)
lossIterations = 50 -- in which interval to report training

-- (5.1) We keep record of training statistics:
-- loss, KLD loss, mean, std and logvar
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

  -- (5.2) One training step, consisting of forward pass
  -- and criterion evaluation and backward pass.
  -- Optimization is then performed by ADAM.
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

  -- Check https://github.com/torch/optim/blob/master/adam.lua
  -- for details on learning rate decay.
  adamState = adamState or {
      learningRate = learningRate,
      momentum = 0,
      learningRateDecay = 0.0001
  }

  -- Returns the new parameters and the objective evaluated
  -- before the update.
  p, f = optim.adam(feval, parameters, adamState)

  -- (5.3) Occasionally, we print the most relevant information
  -- including loss and KLD loss as well as latent code statistics.
  if t%lossIterations == 0 then
    local loss = torch.mean(protocol:narrow(2, 1, 1):narrow(1, t - lossIterations + 1, lossIterations))
    local KLDLoss = torch.mean(protocol:narrow(2, 2, 1):narrow(1, t - lossIterations + 1, lossIterations))
    local mean = torch.mean(protocol:narrow(2, 3, 1):narrow(1, t - lossIterations + 1, lossIterations))
    local std = torch.mean(protocol:narrow(2, 4, 1):narrow(1, t - lossIterations + 1, lossIterations))
    local logvar = torch.mean(protocol:narrow(2, 5, 1):narrow(1, t - lossIterations + 1, lossIterations))
    print('[Training] ' .. t .. '/' .. iterations .. ': ' .. loss .. ' | ' .. KLDLoss .. ' | ' .. mean .. ' | ' .. std .. ' | ' .. logvar)
  end
end

-- (6) For visualization, interpolations are generated;
-- in this case this is easy as the code is two-dimensional.
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