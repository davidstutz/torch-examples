-- Adapted from https://github.com/e-lab/torch-toolbox/blob/master/Weight-init/weight-init.lua.
require("nn")

-- (1) Define several initialization schemes; these compute, given
-- the number of input and output units the standard deviation
-- to be used for Module.reset.

--- Initialization scheme introduced in
-- "Efficient backprop", Yann Lecun, 1998
-- @param fanIn number of input units
-- @param fanOut number of output units
-- @return standard deviation to use
local function initHeuristic(fanIn, fanOut)
  return math.sqrt(1/(3*fanIn))
end

--- Initialization scheme introduced in
-- "Understanding the difficulty of training deep feedforward neural networks", Xavier Glorot, 2010
-- @param fanIn number of input units
-- @param fanOut number of output units
-- @return standard deviation to use
local function initXavier(fanIn, fanOut)
  return math.sqrt(2/(fanIn + fanOut))
end

--- Initialization scheme introduced in
-- "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification", Kaiming He, 2015
-- @param fanIn number of input units
-- @param fanOut number of output units
-- @return standard deviation to use
local function initKaiming(fanIn, fanOut)
  return math.sqrt(4/(fanIn + fanOut))
end

--- Use the given method to initialize all layers.
-- @param model model to initialize
-- @param methodName method to use
-- @return model
local function init(model, methodName)
  methodName = methodName or 'xavier'
  
  local method = nil
  if methodName == 'heuristic' then
    method = initHeuristic
  elseif methodName == 'xavier' then
    method = initXavier
  elseif methodName == 'kaiming' then
    method = initKaiming
  else
    assert(false)
  end

  -- (2) Go through all modules and apply the chosen initialization scheme
  -- to layers with weights and biases.
  for i = 1, #model.modules do
    local m = model.modules[i]
    if m.__typename == 'nn.SpatialConvolution' then
      m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
    elseif m.__typename == 'nn.SpatialConvolutionMM' then
      m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
    elseif m.__typename == 'nn.Linear' then
      m:reset(method(m.weight:size(2), m.weight:size(1))) 
    end

    if m.bias then
      m.bias:zero()
    end
  end
  
  return model
end

return init