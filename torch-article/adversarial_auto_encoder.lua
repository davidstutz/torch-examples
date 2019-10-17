-- Trying to implement an adversarial auto-encoder.

require('nngraph')
require('torch')
require('cutorch')
require('nn')
require('cunn')
require('optim')
require('image')
require('lfs')

package.path = package.path .. ";" .. lfs.currentdir() .. '?.lua'
init = require('init')

-- Interfers with image
--torch.setdefaulttensortype('torch.CudaTensor')

--- RMSProp on the given objective.
-- @param opfunc objective to optimize
-- @param x parameters to optimize
-- @param config configuration
function rmsprop(opfunc, x, config)
    -- (0) Check the configuration.
    assert(config)
    assert(config.learningRate)
    assert(config.momentum)
    assert(config.learningRateDecay)
    assert(config.numUpdates)
    assert(config.optimize ~= nil)

    -- (1) Update learning rate.
    if config.numUpdates%config.decayStep == 0 then
      config.learningRate = config.learningRate*config.learningRateDecay
    end

    local lr = config.learningRate
    local alpha = config.alpha or 0.9
    local epsilon = config.epsilon or 1e-8

    -- (2) Evaluate f(x) and df/dx.
    local fx, dfdx = opfunc(x)

    -- If the model is set to optimize, update parameters:
    if config.optimize == true then
        -- (3) Initialize mean square values and square gradient storage.
        if not config.m then
          config.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
          config.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
        end

        -- (4) Calculate new (leaky) mean squared values.
        config.m:mul(alpha)
        config.m:addcmul(1.0 - alpha, dfdx, dfdx)

        -- (5) Perform update.
        config.tmp:sqrt(config.m):add(epsilon)
        x:addcdiv(-lr, dfdx, config.tmp)
    end
    config.numUpdates = config.numUpdates + 1

    -- return x*, f(x) before optimization
    return x, {fx, dfdx}
end

--- Train a GAN, i.e. the generator and discriminator.
-- @param data training data
function train(data)
  model_D:training()
  model_Dec:training()
  model_EncDec:training()

  epoch = epoch or 1
  local N = data:size(1)
  local dataBatchSize = batchSize/2

  -- Do one epoch.
  for t = 1, math.floor(N/dataBatchSize) do

    -- inputs holds the true inputs for reconstruction.
    -- targets will be adapted to the targets in each stage.
    -- codes will hold half true half random codes.
    local inputs = torch.Tensor(batchSize, 1, H, W)
    local outputs = torch.Tensor(dataBatchSize, 1, H, W)
    local targets = torch.Tensor(batchSize)
    local codes = torch.Tensor(batchSize, cD)

    inputs = inputs:cuda()
    outputs = outputs:cuda()
    targets = targets:cuda()
    codes = codes:cuda()

    --- Function to evaluate the discriminator.
    -- @param x parameters
    -- @return function value and derivative
    local fevalD = function(x)
      collectgarbage()

      if x ~= parameters_D then
        parameters_D:copy(x)
      end

      gradParameters_D:zero()

      -- Foward pass.
      local preds = model_D:forward(inputs)

      -- Compute error on real and fake data.
      -- Remember that the first dataBatchSize elements are real.
      err_R = criterion:forward(preds:narrow(1, 1, dataBatchSize), targets:narrow(1, 1, dataBatchSize))
      err_F = criterion:forward(preds:narrow(1, dataBatchSize + 1,dataBatchSize), targets:narrow(1, dataBatchSize + 1, dataBatchSize))
      err = criterion:forward(preds, targets)
      --print(err_R, err_F)

      -- err_R, err_F will be roughly 0.7 if training gets stuck, so the margin should be
      -- a bit above 0.3!
      local margin = 0.31
      sgdState_D.optimize = true
      sgdState_Dec.optimize = true

      if err_F < margin or err_R < margin then
         sgdState_D.optimize = false
      end
      if err_F > (1.0 - margin) or err_R > (1.0 - margin) then
         sgdState_Dec.optimize = false
      end
      if math.abs(err - 0.5) < 0.01 then
        sgdState_Dec.optimize = false
      end

      -- Avoid a deadlock.
      -- Note that deadlock means both not optimizing and not challening
      -- each other (i.e. if both are really bad.
      if sgdState_Dec.optimize == false and sgdState_D.optimize == false then
        local r = math.random()
        if r > 0.5 then
          sgdState_Dec.optimize = true
        else
          sgdState_D.optimize = true
        end
      end

      local f = criterion:forward(preds, targets)

      local df_do = criterion:backward(preds, targets)
      model_D:backward(inputs, df_do)

      -- L_1 or L_2 penalties/regularizers.
      if coefL1 > 0 or coefL2 > 0 then
        -- Loss:
        f = f + coefL1 * torch.norm(parameters_D, 1)
        f = f + coefL2 * torch.norm(parameters_D, 2)^2/2

        -- Gradients:
        gradParameters_D:add(torch.sign(parameters_D):mul(coefL1) + parameters_D:clone():mul(coefL2))
      end

      --print('[Training][' .. t .. '] D ' .. f .. '(' .. gradParameters_D:norm() .. ')')
      return f, gradParameters_D
    end

    --- Function to evaluate the auto-encoder.
    -- @param x parameters
    -- @return function value and derivative
    local fevalEncDec = function(x)
      collectgarbage()

      if x ~= parameters_EncDec then
        parameters_EncDec:copy(x)
      end

      gradParameters_EncDec:zero()

      -- Foward pass.
      local preds = model_EncDec:forward(inputs)
      local f = criterion:forward(preds, outputs)

      --  Backward pass.
      local df = criterion:backward(preds, outputs)
      model_EncDec:backward(inputs, df)

      --print('[Training][' .. t .. '] EncDec ' .. f .. '(' .. gradParameters_EncDec:norm() .. ')')
      return f, gradParameters_EncDec
    end

    --- Function to evaluate the decoder/generator.
    -- @param x parameters
    -- @return function value and derivative
    local fevalDec = function(x)
      collectgarbage()

      if x ~= parameters_Dec then
        parameters_Dec:copy(x)
      end

      gradParameters_Dec:zero()

      -- Foward pass.
      local samples = model_Dec:forward(codes)
      local preds = model_D:forward(samples)
      local f = criterion:forward(preds, targets)

      --  Backward pass.
      local df_samples = criterion:backward(preds, targets)
      model_D:backward(samples, df_samples)

      local df = model_D.modules[1].gradInput
      model_Dec:backward(codes, df)

      --print('[Training][' .. t .. '] Dec ' .. f .. '(' .. gradParameters_Dec:norm() .. ')')
      return f, gradParameters_Dec
    end

    ----------------------------------------------------------------------
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))).
    -- Get half a minibatch of real, half fake.
    for k = 1, K do

      -- (1.1. Real data; first fetch random samples, then reconstruct them
      -- take the reconstructions as inputs.
      for i = 1, dataBatchSize do
        local sample = data[math.random(N)]
        inputs[i] = sample:clone()
      end


      local samples = model_EncDec:forward(inputs:narrow(1, 1, dataBatchSize))
      local j = 1
      for i = 1, dataBatchSize do
        inputs[j] = samples[i]:clone()
        j = j + 1
      end

      -- (1.2) Real targets.
      targets[{{1,dataBatchSize}}]:fill(1)

      -- (1.3) Sampled data.
      -- Note not to sample from a normal distribution instead!
      codes:uniform(-1, 1)
      samples = model_Dec:forward(codes:narrow(1, 1, dataBatchSize))
      for i = 1, dataBatchSize do
        inputs[j] = samples[i]:clone()
        j = j + 1
      end

      -- (1.4) Sampled targets.
      targets[{{dataBatchSize + 1, batchSize}}]:fill(0)

      p, f_D = rmsprop(fevalD, parameters_D, sgdState_D)

      if sgdState_D.optimize then
        optimizedCount_D = optimizedCount_D + 1
      end
    end

    ----------------------------------------------------------------------
    -- (2) Update auto-encoder network: minimize reconstruction loss
    -- Get a full mini-batch of data samples.
    for l = 1, L do
      for i = 1, batchSize do
        local sample = data[math.random(N)]
        inputs[j] = sample:clone()
      end

      outputs = inputs:clone()

      p, f_EncDec = rmsprop(fevalEncDec, parameters_EncDec, sgdState_EncDec)
      optimizedCount_EncDec = optimizedCount_EncDec + 1
    end

    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z))).
    for m = 1, M do
      codes:uniform(-1, 1)
      targets:fill(1)
      p, f_Dec = rmsprop(fevalDec, parameters_Dec, sgdState_Dec)

      if sgdState_Dec.optimize then
        optimizedCount_Dec = optimizedCount_Dec + 1
      end
    end

    smoothedLoss_D = smoothedLoss_D + f_D[1]
    smoothedLoss_Dec = smoothedLoss_Dec + f_Dec[1]
    smoothedLoss_EncDec = smoothedLoss_EncDec + f_EncDec[1]
    smoothedGradParameters_D = smoothedGradParameters_D + torch.norm(f_D[2], 2)
    smoothedGradParameters_Dec = smoothedGradParameters_Dec + torch.norm(f_Dec[2], 2)
    smoothedGradParameters_EncDec = smoothedGradParameters_EncDec + torch.norm(f_EncDec[2], 2)
    smoothedCount = smoothedCount + 1

    if t%50 == 0 then
      print('[Training] EncDec: ' .. smoothedLoss_EncDec/smoothedCount .. ' (' .. smoothedGradParameters_EncDec/smoothedCount .. ', ' .. optimizedCount_EncDec/smoothedCount .. ') '
        .. ' Dec: ' .. smoothedLoss_Dec/smoothedCount .. ' (' .. smoothedGradParameters_Dec/smoothedCount .. ', ' .. optimizedCount_Dec/(K*smoothedCount) .. ')'
      .. ' D: ' .. smoothedLoss_D/smoothedCount .. ' (' .. smoothedGradParameters_D/smoothedCount .. ', ' .. optimizedCount_D/(K*smoothedCount) .. ')')

      optimizedCount_D = 0
      optimizedCount_Dec = 0
      optimizedCount_EncDec = 0
      smoothedGradParameters_D = 0
      smoothedGradParameters_Dec = 0
      smoothedGradParameters_EncDec = 0
      smoothedLoss_D = 0
      smoothedLoss_Dec = 0
      smoothedLoss_EncDec = 0
      smoothedCount = 0
    end
  end

  -- TODO save snapshots!
  --torch.save(filename, {D = model_D, G = model_G, opt = opt})

  epoch = epoch + 1
end

--- Test the GAN.
-- @param data real data to test on
function test(data)

  -- Test reconstruction.
  local preds = model_EncDec:forward(data)
  print('[Testing] Reconstruction (abs) ' .. torch.sum(torch.abs(preds - data))/(data:size(1)*data:size(3)*data:size(4)))

  -- Write first few data images and reconstructions to images.
  for i = 1, 100 do
    image.save('results/' .. i .. '.png', data[i][1])
    image.save('results/' .. i .. '_rec.png', preds[i][1])
  end

  -- Test discriminator.
  preds = model_D:forward(preds)
  preds[preds:gt(0.5)] = 1
  preds[preds:lt(0.5)] = 0
  print('[Testing] Discriminator (accuracy) ' .. torch.sum(preds)/data:size(1))
end

-- Training parameters.
K = 4 -- Number of D rounds per iteration ...
L = 2 -- Number of EncDec rounds per iterion ...
M = 1 -- Number of Dec rounds per iteration ...
coefL1 = 0
coefL2 = 0
batchSize = 8
momentum = 0
learningRate = 0.005

-- Data parameters.
H = 16
W = 16
rH = 8
rW = 8
cD = 2
N = 5000

-- Fix random seed.
--torch.manualSeed(1)

-- Generate rectangle data.
trainData = torch.Tensor(N, 1, H, W):fill(0)
for i = 1, N do
  local h = torch.random(2, rH)
  local w = torch.random(2, rW)
  local aH = torch.random(1, H - h)
  local aW = torch.random(1, W - w)
  trainData[i][1]:sub(aH, aH + h, aW, aW + w):fill(1)
end

testData = torch.Tensor(N, 1, H, W):fill(0)
for i = 1, math.floor(0.1*N) do
  local h = torch.random(2, rH)
  local w = torch.random(2, rW)
  local aH = torch.random(1, H - h)
  local aW = torch.random(1, W - w)
  testData[i][1]:sub(aH, aH + h, aW, aW + w):fill(1)
end

trainData = trainData:cuda()
testData = testData:cuda()

-- Discriminator network.
model_D = nn.Sequential()
model_D:add(nn.SpatialConvolution(1, 4, 3, 3, 1, 1, 1, 1))
model_D:add(nn.SpatialBatchNormalization(4))
model_D:add(nn.ReLU())
model_D:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
model_D:add(nn.SpatialConvolution(4, 8, 3, 3, 1, 1, 1, 1))
model_D:add(nn.SpatialBatchNormalization(8))
model_D:add(nn.ReLU())
model_D:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0 ,0))
model_D:add(nn.SpatialConvolution(8, 16, 3, 3, 1, 1, 1, 1))
model_D:add(nn.SpatialBatchNormalization(16))
model_D:add(nn.ReLU())
model_D:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
model_D:add(nn.View(4*16))
model_D:add(nn.Linear(4*16, 1))
model_D:add(nn.Sigmoid())
--model_D = init(model_D)

-- Encoder.
model_Enc = nn.Sequential()
model_Enc:add(nn.SpatialConvolution(1, 4, 3, 3, 1, 1, 1, 1))
model_Enc:add(nn.ReLU())
model_Enc:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
model_Enc:add(nn.SpatialConvolution(4, 8, 3, 3, 1, 1, 1, 1))
model_Enc:add(nn.ReLU())
model_Enc:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0 ,0))
model_Enc:add(nn.SpatialConvolution(8, 16, 3, 3, 1, 1, 1, 1))
model_Enc:add(nn.ReLU())
model_Enc:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))
model_Enc:add(nn.View(4*16))
model_Enc:add(nn.Linear(4*16, cD))
--model_Enc = init(model_Enc)

-- Decoder.
model_Dec = nn.Sequential()
model_Dec:add(nn.Linear(cD, 4*16))
model_Dec:add(nn.View(16, 2, 2))
model_Dec:add(nn.SpatialUpSamplingNearest(2))
model_Dec:add(nn.SpatialConvolution(16, 8, 3, 3, 1, 1, 1, 1))
model_Dec:add(nn.ReLU())
model_Dec:add(nn.SpatialUpSamplingNearest(2))
model_Dec:add(nn.SpatialConvolution(8, 4, 3, 3, 1, 1, 1, 1))
model_Dec:add(nn.ReLU())
model_Dec:add(nn.SpatialUpSamplingNearest(2))
model_Dec:add(nn.SpatialConvolution(4, 1, 3, 3, 1, 1, 1, 1))
model_Dec:add(nn.Sigmoid())
--model_Dec = init(model_Dec)

-- Auto-encoder.
model_EncDec = nn.Sequential()
model_EncDec:add(model_Enc:clone('weight', 'bias', 'gradWeight', 'gradBias'))
model_EncDec:add(model_Dec:clone('weight', 'bias', 'gradWeight', 'gradBias'))

model_D = model_D:cuda()
model_Dec = model_Dec:cuda()
model_EncDec = model_EncDec:cuda()

-- Loss function: negative log-likelihood.
criterion = nn.BCECriterion()
criterion = criterion:cuda()

-- Retrieve parameters and gradients.
parameters_D, gradParameters_D = model_D:getParameters()
parameters_Dec, gradParameters_Dec = model_Dec:getParameters()
parameters_EncDec, gradParameters_EncDec = model_EncDec:getParameters()

-- Training parameters
sgdState_D = {
  learningRate = learningRate,
  momentum = momentum,
  learningRateDecay = 0.95,
  decayStep = math.floor(N/batchSize), -- Update learning rate each epoch.
  optimize = true,
  numUpdates = 0
}

sgdState_Dec = {
  learningRate = learningRate,
  momentum = momentum,
  learningRateDecay = 0.95,
  decayStep = math.floor(N/batchSize), -- Update learning rate each epoch.
  optimize = true,
  numUpdates = 0
}

sgdState_EncDec = {
  learningRate = learningRate,
  momentum = momentum,
  learningRateDecay = 0.95,
  decayStep = math.floor(N/batchSize), -- Update learning rate each epoch.
  optimize = true,
  numUpdates = 0
}

-- For printing smoothed values.
smoothedGradParameters_D = 0
smoothedGradParameters_Dec = 0
smoothedGradParameters_EncDec = 0
smoothedLoss_D = 0
smoothedLoss_Dec = 0
smoothedLoss_EncDec = 0
smoothedCount = 0

-- Counts optimization to see how often generator and discriminator
-- get updates.
optimizedCount_D = 0
optimizedCount_Dec = 0
optimizedCount_EncDec = 0

while true do
  train(trainData)
  test(testData)

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(learningRate*0.99^epoch, 0.000001)
  sgdState_Dec.momentum = math.min(sgdState_Dec.momentum + 0.0008, 0.7)
  sgdState_Dec.learningRate = math.max(learningRate*0.99^epoch, 0.000001)
  sgdState_EncDec.momentum = math.min(sgdState_EncDec.momentum + 0.0008, 0.7)
  sgdState_EncDec.learningRate = math.max(learningRate*0.99^epoch, 0.000001)
end