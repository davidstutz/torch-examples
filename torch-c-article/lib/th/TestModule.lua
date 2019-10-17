require('torch')
require('nn')

--- @class TestModule
local TestModule, TestModuleParent = torch.class('nn.TestModule', 'nn.Module')

--- Initialize.
function TestModule:__init(threshold)
  self.threshold = threshold or 0.5
end

--- Compute forward pass, i.e. threshold to 1 at 0.1.
-- @param input layer input
-- @param output
function TestModule:updateOutput(input)
  assert(input:type() == 'torch.FloatTensor')
  assert(lib.cpu)

  self.output = input:clone()
  lib.cpu.test_module_updateOutput(5, input:size():data(), input:data(), self.output:data())
  return self.output
end

--- Compute the backward pass.
-- @param input original input
-- @param gradOutput gradients of top layer
-- @return gradients with respect to input
function TestModule:updateGradInput(input, gradOutput)
  assert(input:type() == 'torch.FloatTensor')
  assert(lib.cpu)

  self.gradInput = gradOutput:clone()
  lib.cpu.test_module_updateGradInput(5, input:size():data(), input:data(), gradOutput:data(), self.output:data())
  return self.gradInput
end