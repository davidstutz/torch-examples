-- Convolutional variational auto-encoder.

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

  -- (2.1) Note that the samples assumes CUDA training;
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
N = 10000

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

-- (4) The encoder consists of several linear layers followed by
-- the Kullback Leibler loss, the samples and the docoder; the decoder
-- mirrors the encoder.
-- (4.1) The encoder:
encoder = nn.Sequential()
encoder:add(nn.SpatialConvolutionMM(1, 8, 3, 3, 1, 1, 1, 1))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))
encoder:add(nn.SpatialConvolutionMM(8, 16, 3, 3, 1, 1, 1, 1))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))
encoder:add(nn.SpatialConvolutionMM(16, 32, 3, 3, 1, 1, 1, 1))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))

hidden = H/8*W/8*32
code = 2
encoder:add(nn.View(hidden))
meanLogVar = nn.ConcatTable()
meanLogVar:add(nn.Linear(hidden, code)) -- Mean of the hidden code.
meanLogVar:add(nn.Linear(hidden, code)) -- Variance of the hidden code (diagonal variance matrix).
encoder:add(meanLogVar)

-- (4.2) The decoder:
decoder = nn.Sequential()
decoder:add(nn.Linear(code, hidden))
decoder:add(nn.ReLU(true))

decoder:add(nn.View(32, H/8, W/8))
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(nn.SpatialConvolutionMM(32, 16, 3, 3, 1, 1, 1, 1))
decoder:add(nn.ReLU(true))
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(nn.SpatialConvolutionMM(16, 8, 3, 3, 1, 1, 1, 1))
decoder:add(nn.ReLU(true))
decoder:add(nn.SpatialUpSamplingNearest(2))
decoder:add(nn.SpatialConvolutionMM(8, 1, 3, 3, 1, 1, 1, 1))
decoder:add(nn.Sigmoid(true))

-- (4.3) The full model, i.e encoder followed by the Kullback Leibler
-- divergence and the reparameterization trick sampler.
-- Note that the Kullback Leibler divergence is weighted.
-- The weight has crucial influence on the results.
model = nn.Sequential()
model:add(encoder)
KLD = nn.KullbackLeiblerDivergence()
model:add(KLD)
model:add(nn.ReparameterizationSampler())
model:add(decoder)
model = model:cuda()

-- (4.4) As criterion, a binary cross entropy criterion is used (as
-- for classification), note that this is also discussed in the paper.
criterion = nn.BCECriterion()
criterion.sizeAverage = false
criterion = criterion:cuda()

parameters, gradParameters = model:getParameters()
parameters = parameters:cuda()
gradParameters = gradParameters:cuda()

-- (5) Training proceeds as for regular networks.
-- The BCE loss and the Kullback Leibler loss are monitored
-- separately.
batchSize = 8
learningRate = 0.001
iterations = math.floor(5*N/batchSize)

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

    -- Estimate df/dW.
    local df_do = criterion:backward(pred, output)
    model:backward(input, df_do)

    -- return f and df/dX
    return f, gradParameters
  end

  adamState = adamState or {
      learningRate = learningRate,
      momentum = 0,
      learningRateDecay = 5e-7
  }

  -- Returns the new parameters and the objective evaluated
  -- before the update.
  p, f = optim.adam(feval, parameters, adamState)

  print('[Training] ' .. t .. ': ' .. f[1] .. ' ' .. KLD.loss)
end

-- (6) For verifying the code distribution, the mean and standard deviation is printed.
inputs = inputs:cuda()
outputs = encoder:forward(inputs)
outputs[1] = outputs[1]:cuda()
outputs[2] = outputs[2]:float()
print(torch.mean(outputs[1]), torch.std(outputs[1]))

-- (7) For visualization, interpolations are generated;
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