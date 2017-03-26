-- A more complex initialization module which allows non-uniform initialization 
-- not using the layers' reset method.
require("nn")

-- (1) The initialization methods now take the tensor
-- to be initialized (weights or biases) and an optional
-- value in addition to fan in and fan out.

--- Initialize a tensor with a fixed value.
-- @param tensor tensor to initialize
-- @param fanIn number of input units
-- @param fanOut number of output units
-- @param value value to initialize with
local function initFixed(tensor, fanIn, fanOut, value)
  if tensor then
    tensor:fill(value)
  end
end

--- Uniform initialization.
-- @param tensor tensor to initialize
-- @param fanIn number of input units
-- @param fanOut number of output units
-- @param value value to use as range for uniform intialization
local function initUniform(tensor, fanIn, fanOut, value)
  if tensor then
    tensor:uniform(-value, value)
  end
end

--- Initialize a tensor according to normal distribution.
-- @param tensor tensor to initialize
-- @param fanIn number of input units
-- @param fanOut number of output units
-- @param value value to initialize with
local function initNormal(tensor, fanIn, fanOut, value)
  if tensor then
    tensor.normal(0, value)
  end
end

--- Initialization scheme introduced in
-- "Efficient backprop", Yann Lecun, 1998
-- @param tensor tensor to initialize
-- @param fanIn number of input units
-- @param fanOut number of output units
-- @param value not used
local function initHeuristic(tensor, fanIn, fanOut, value)
  local std = math.sqrt(1/(3*fanIn))
  std = std * math.sqrt(3)
  
  if tensor then
    tensor:uniform(-std, std)
  end
end

--- Initialization scheme introduced in
-- "Understanding the difficulty of training deep feedforward neural networks", Xavier Glorot, 2010
-- @param tensor tensor to initialize
-- @param fanIn number of input units
-- @param fanOut number of output units
-- @param value not used
local function initXavier(tensor, fanIn, fanOut, value)
  local std = math.sqrt(2/(fanIn + fanOut))
  std = std * math.sqrt(3)
  
  if tensor then
    tensor:uniform(-std, std)
  end
end

--- Initialization scheme introduced in
-- @param tensor tensor to initialize
-- @param fanIn number of input units
-- @param fanOut number of output units
-- @param value not used
local function initKaiming(tensor, fanIn, fanOut, value)
  local std = math.sqrt(4/(fanIn + fanOut))
  std = std * math.sqrt(3)
  
  if tensor then
    tensor:uniform(-std, std)
  end
end

--- Get the init function by its name.
-- @param name name of the function
-- @return the function
local function getMethodByName(name)
  if name == 'fixed' then
    return initFixed
  elseif name == 'uniform' then
    return initUniform
  elseif name == 'normal' then
    return initNormal
  elseif name == 'heursitic' then
    return initHeuristic
  elseif name == 'xavier' then
    return initXavier
  elseif name == 'kaiming' then
    return initKaiming
  else
    assert(false)
  end
end

--- Use the given method to initialize all layers.
-- @param model model to initialize
-- @param methodName method to use
-- @return model
local function init(model, weightsMethodName, weightsValue, biasMethodName, biasValue)
  
  local weightsValue = weightsValue or 0.05
  local biasMethodName = biasMethodName or 'fixed'
  local biasValue = biasValue or 0.0
  
  -- (3) Different initialization schemes for weights and biases.
  local weightsMethod = getMethodByName(weightsMethodName)
  local biasMethod = getMethodByName(biasMethodName)

  -- (2) Loop over all modules and separately initialize weights and biases.
  -- Depending on the chosen initialization method, the optional value can or cannot be used.
  for i = 1, #model.modules do
    local m = model.modules[i]
    if m.__typename == 'nn.SpatialConvolution' then
      weightsMethod(m.weight, m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW, weightsValue)
      biasMethod(m.bias, m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW, biasValue)
    elseif m.__typename == 'nn.SpatialConvolutionMM' then
      weightsMethod(m.weight, m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW, weightsValue)
      biasMethod(m.bias, m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW, biasValue)
    elseif m.__typename == 'nn.Linear' then
      weightsMethod(m.weight, m.weight:size(2), m.weight:size(1), weightsValue)
      biasMethod(m.bias, m.weight:size(2), m.weight:size(1), biasValue)
    end
  end
  
  return model
end

return init