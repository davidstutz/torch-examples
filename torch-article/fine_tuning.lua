-- This module implements several "fine-tuning" layers, i.e. general
-- layers where the accGradParameters function is empty as discussed
-- here: https://groups.google.com/forum/#!topic/torch7/S8hWQtEIkxg

require('nn')

--- @class LinearFT
local LinearFT, LinearFTParent = torch.class('nn.LinearFT', 'nn.Linear')

--- Required constructor.
-- @param inputSize number of input units
-- @param outputSize number of output units
-- @param bias whether to use a bias
function LinearFT:__init(inputSize, outputSize, bias)
  LinearFTParent.__init(self, inputSize, outputSize, bias)
end

--- Avoids accumulating gradient in order to fix the parameters
-- for fine-tuning.
function LinearFT:accGradParameters(input, gradOutput, scale)
  -- Nothing!
end

local LinearDead, LinearDeadParent = torch.class('nn.LinearDead', 'nn.LinearFT')

--- Required constructor.
-- @param inputSize number of input units
-- @param outputSize number of output units
-- @param bias whether to use a bias
function LinearDead:__init(inputSize, outputSize, bias)
  LinearDeadParent.__init(self, inputSize, outputSize, bias)
end

-- Avoids passing the gradients down.
function LinearDead:updateGradInput(input, gradOutput)
   -- Nothing!
end