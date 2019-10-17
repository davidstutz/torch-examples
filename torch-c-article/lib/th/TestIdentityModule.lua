require('torch')
require('nn')

--- @class TestIdentityModule
local TestIdentityModule, TestIdentityModuleParent = torch.class('nn.TestIdentityModule', 'nn.Module')

--- Initialize.
function TestIdentityModule:__init()

end

--- Compute forward pass.
-- @param input layer input
-- @param output
function TestIdentityModule:updateOutput(input)
  self.output = input:clone()

  if input:type() == 'torch.FloatTensor' then
    assert(lib.cpu)
    lib.cpu.test_identity_module_updateOutput(5, input:size():data(), input:data(), self.output:data())
  elseif input:type() == 'torch.CudaTensor' then
    assert(lib.gpu)
    lib.gpu.test_identity_module_updateOutput(5, input:size():data(), input:data(), self.output:data())
  else
    assert(false)
  end

  return self.output
end

--- Compute the backward pass.
-- @param input original input
-- @param gradOutput gradients of top layer
-- @return gradients with respect to input
function TestIdentityModule:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput:clone()

  if input:type() == 'torch.FloatTensor' then
    assert(lib.cpu)
    lib.cpu.test_identity_module_updateGradInput(5, input:size():data(), input:data(), gradOutput:data(), self.gradInput:data())
  elseif input:type() == 'torch.CudaTensor' then
    assert(lib.gpu)
    lib.gpu.test_identity_module_updateGradInput(5, input:size():data(), input:data(), gradOutput:data(), self.gradInput:data())
  else
    assert(false)
  end

  return self.gradInput
end