-- Generative adversarial network.

require('nngraph')
require('torch')
require('nn')
require('cunn')
require('optim')
require('randomkit')

-- (1) Simple wrapper around optim.rmsprop allowing to turn
-- of optmization for either the descriminator or generator using
-- config.optimize.
--- Wrapper around RMSProp checking config.optimize.
-- @param opfunc objective to optimize
-- @param x parameters to optimize
-- @param config configuration
function rmsprop(opfunc, x, config)
    if config.optimize then
      return optim.rmsprop(opfunc, x, config)
    else
      local fx, dfdx = opfunc(x)
      return x, {fx}
    end
end

-- (2) Data generation; the example will try to learn a one-dimension
-- Gaussian with the given mean and standard deviation
D = 1
nD = 2
N = 10000
mean = 3
std = 1

torch.setdefaulttensortype('torch.CudaTensor')
trainData = torch.Tensor(N, D):normal(mean, std)
testData = torch.Tensor(N, D):normal(mean, std)
print('[Data] mean ' .. torch.mean(trainData) .. ' std ' .. torch.std(trainData))

-- (3) The discriminator network is a simple classification network.
-- Depending on the task and the generator, the discriminator can be
-- made weaker by considering dropout, and made stronger by considering
-- batch normalization or a deeper/wider architecture. This might be necessary
-- if either the discriminator learns to fast, or the generator is too powerful.
-- Discriminator network.
model_D = nn.Sequential()
model_D:add(nn.Linear(D, 4*D))
model_D:add(nn.ReLU(true))
--model_D:add(nn.Dropout())
model_D:add(nn.Linear(4*D, 4*D))
model_D:add(nn.ReLU(true))
--model_D:add(nn.Dropout())
model_D:add(nn.Linear(4*D, 1))
model_D:add(nn.Sigmoid())

-- (4) The generator takes a nD-dimensional noise input and turns it into
-- a one-dimensional value which is supposed to be distributed like a Gaussian.
-- The generator can be made stronger with a deeper architecture or by
-- considering batch normalization, if necessary.
-- Generator network.
model_G = nn.Sequential()
model_G:add(nn.Linear(nD, 4*D))
--model_G:add(nn.BatchNormalization(4*D))
model_G:add(nn.ReLU(true))
--model_G:add(nn.Linear(4*D, 4*D))
--model_G:add(nn.BatchNormalization(4*D))
--model_G:add(nn.ReLU(true))
model_G:add(nn.Linear(4*D, D))

model_D = model_D:cuda()
model_G = model_G:cuda()

criterion = nn.BCECriterion()
criterion = criterion:cuda()

-- Retrieve parameters and gradients.
parameters_D, gradParameters_D = model_D:getParameters()
parameters_G, gradParameters_G = model_G:getParameters()

-- (5) Training will be governed by the regular learning parameters;
-- in addition, K will denote the number the disrcriminator is updated per step.
K = 3
batchSize = 8
momentum = 0
learningRate = 0.005
epochs = 5

sgdState_D = {
  learningRate = learningRate,
  momentum = momentum,
  learningRateDecay = 0.95,
  decayStep = math.floor(N/batchSize),
  optimize = true
}

sgdState_G = {
  learningRate = learningRate,
  momentum = momentum,
  learningRateDecay = 0.95,
  decayStep = math.floor(N/batchSize),
  optimize = true
}

-- For monitoring training.
smoothedLoss_G = 0
smoothedLoss_D = 0
smoothedCount = 0
optimizedCount_G = 0
optimizedCount_D = 0

-- (6) Training.
for t = 1, epochs do

  -- Do one epoch.
  local dataBatchSize = batchSize/2
  for t = 1, math.floor(N/dataBatchSize) do

    -- Random input for data generator.
    local inputs = torch.Tensor(batchSize, D)
    local targets = torch.Tensor(batchSize)
    local noise_inputs = torch.Tensor(batchSize, nD)

    --- Function to evaluate the discriminator.
    -- @param x parameters
    -- @return function value and derivative
    local fevalD = function(x)
      collectgarbage()

      if x ~= parameters_D then
        parameters_D:copy(x)
      end

      gradParameters_D:zero()

      local outputs = model_D:forward(inputs)

      -- Compute error on real and fake data.
      -- Remember that the first dataBatchSize elements are real.
      err_R = criterion:forward(outputs:narrow(1, 1, dataBatchSize), targets:narrow(1, 1, dataBatchSize))
      err_F = criterion:forward(outputs:narrow(1, dataBatchSize + 1,dataBatchSize), targets:narrow(1, dataBatchSize + 1, dataBatchSize))
      err = criterion:forward(outputs, targets)

      -- err_R, err_F will be roughly 0.7 if training gets stuck, so the margin should be
      -- a bit above 0.3!
      local margin = 0.31
      sgdState_D.optimize = true
      sgdState_G.optimize = true

      if err_F < margin or err_R < margin then
         sgdState_D.optimize = false
      end
      if err_F > (1.0 - margin) or err_R > (1.0 - margin) then
         sgdState_G.optimize = false
      end
      if math.abs(err - 0.5) < 0.01 then
        sgdState_G.optimize = false
      end

      -- Avoid a deadlock.
      -- Note that deadlock means both not optimizing and not challening
      -- each other (i.e. if both are really bad.
      if sgdState_G.optimize == false and sgdState_D.optimize == false then
        local r = math.random()
        if r > 0.5 then
          sgdState_G.optimize = true
        else
          sgdState_D.optimize = true
        end
      end

      local f = criterion:forward(outputs, targets)

      local df_do = criterion:backward(outputs, targets)
      model_D:backward(inputs, df_do)

      return f, gradParameters_D
    end

    --- Function to evaluate the generator.
    -- @param x parameters
    -- @return function value and derivative
    local fevalG = function(x)
      collectgarbage()

      if x ~= parameters_G then
        parameters_G:copy(x)
      end

      gradParameters_G:zero()

      local samples = model_G:forward(noise_inputs)
      local outputs = model_D:forward(samples)
      local f = criterion:forward(outputs, targets)

      local df_samples = criterion:backward(outputs, targets)
      model_D:backward(samples, df_samples)

      local df_do = model_D.modules[1].gradInput
      model_G:backward(noise_inputs, df_do)

      return f, gradParameters_G
    end

    -- (6.1) The discriminator is updated K times; each time, a batch consisting of real
    -- and fake data is chosen and the discriminator is supposed to distinguish them.
    for k = 1, K do

      -- (6.1.1) Real data chosen randomly from the trianing set.
      local j = 1
      for i = t, math.min(t + dataBatchSize - 1, N) do
        local sample = trainData[math.random(N)]
        inputs[j] = sample:clone()
        j = j + 1
      end

      targets[{{1,dataBatchSize}}]:fill(1)

      -- (6.1.2) Fake data comes form a uniform distribution.
      -- Instead, other distributions different enough from the target distributions could
      -- be chosen.
      local samples = model_G:forward(torch.Tensor(dataBatchSize, nD):uniform(-1, 1))
      for i = 1, dataBatchSize do
        inputs[j] = samples[i]:clone()
        j = j + 1
      end

      targets[{{dataBatchSize + 1, batchSize}}]:fill(0)

      -- (6.1.3) fevalD additionally decides whether to update the discriminator or
      -- generator. Details can be found in fevalD; explained in short,
      -- the discriminators error on real and fake data is considered.
      -- If the discriminator is too strong, it will not be updated further,
      -- if it is too weak, the generator is not updated further.
      p, f_D = rmsprop(fevalD, parameters_D, sgdState_D)

      if sgdState_D.optimize then
        optimizedCount_D = optimizedCount_D + 1
      end
    end

    -- (6.2) Update the generator, giving noise as input;
    -- fevalG encodes the objective, i.e. the generator is trained to fool
    -- the discriminator.
    noise_inputs:uniform(-1, 1)
    targets:fill(1)
    p, f_G = rmsprop(fevalG, parameters_G, sgdState_G)

    if sgdState_G.optimize then
      optimizedCount_G = optimizedCount_G + 1
    end

    smoothedLoss_G = smoothedLoss_G + f_G[1]
    smoothedLoss_D = smoothedLoss_D + f_D[1]
    smoothedCount = smoothedCount + 1

    if t%50 == 0 then
      print('[Training] G: ' .. smoothedLoss_G/smoothedCount .. ' (' .. optimizedCount_G/smoothedCount .. ') '
        .. 'D: ' .. smoothedLoss_D/smoothedCount .. ' (' .. optimizedCount_D/(K*smoothedCount) .. ')')

      optimizedCount_G = 0
      optimizedCount_D = 0
      smoothedLoss_G = 0
      smoothedLoss_D = 0
      smoothedCount = 0
    end
  end

  -- (6.4) Both the generator and the discriminator are tested.
  -- The discriminator is tested using real and fake data, for testing the generator,
  -- the mean and standard deviation of the predicted outputs are inspected.
  local noise_inputs = torch.Tensor(N, nD):uniform(-1, 1)
  local samples = model_G:forward(noise_inputs)

  local preds = model_D:forward(samples)
  preds[preds:lt(0.5)] = 0
  preds[preds:gt(0.5)] = 1
  local accuracy = 1 - torch.sum(preds)/N

  preds = model_D:forward(testData)
  preds[preds:lt(0.5)] = 0
  preds[preds:gt(0.5)] = 1
  accuracy = 0.5*accuracy + 0.5*torch.sum(preds)/N

  print('[Testing] error (D): ' .. accuracy .. ', mean (G): ' .. torch.mean(samples) .. ', std (G): ' .. torch.std(samples))
  
  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(learningRate*0.99^t, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
  sgdState_G.learningRate = math.max(learningRate*0.99^t, 0.000001)
end