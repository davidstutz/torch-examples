-- Custom absolute criterion.

require('torch')
require('nn')

-- (1) Small helper function to compute the normalization used
-- in CustomAbsCriterion.
--- Compute the product of elements in a storage.
-- @param storage storage to compute product of
-- @return product of all dimensions
local function storageProd(storage)
  local prod = 1
  for i = 1, #storage do
    prod = prod * storage[i]
  end
  return prod
end

-- (2) Extend nn.Criterion, the newly created criterion is called
-- CustomAbsCriterion and accessible as nn.CustomAbsCriterion after
-- requiring this file.
--- @class CustomAbsCriterion
local CustomAbsCriterion, parent = torch.class('nn.CustomAbsCriterion', 'nn.Criterion')

--- Initialize the criterion.
function CustomAbsCriterion:__init()
   parent.__init(self)
end

-- (3) The forward pass of the criterion, i.e.
-- given inputs and targets, compute the loss.
--- Update/compute output given input and target.
-- @param input input computed by the network
-- @param target target to compute loss on
function CustomAbsCriterion:updateOutput(input, target)
  local norm = storageProd(#input)
  self.output = 1/norm*torch.sum(torch.abs(input - target))
  return self.output
end

-- (4) The backward pass of the criterion, i.e.
-- given original inputs and targets, compute the
-- derivative with respect to the inputs.
--- Update the gradients with respect to the input.
-- @param input input computed by the network
-- @param target target to compute loss on
function CustomAbsCriterion:updateGradInput(input, target)
  self.gradInput:resizeAs(input)
  local difference = input - target
  local norm = storageProd(#input)
  self.gradInput[difference:lt(0)] = -1/norm
  self.gradInput[difference:gt(0)] = 1/norm
  return self.gradInput
end