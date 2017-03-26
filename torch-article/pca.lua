-- (1) Define the layer, extend from nn.Module.
-- The layer will be accessible via nn.PCA.
--- @class PCA
local PCA, PCAParent = torch.class('nn.PCA', 'nn.Module')

-- (2) The constructor can take arbitrary input parameters.
--- Required constructor.
-- @param W weight matrix
-- @param b bias vector
function PCA:__init(W, b)
  PCAParent.__init(self)
  -- (2.1) If layers have parameters, they should use self.weight and self.bias.
  -- Then the parameters are accessible via :getParameters() and the usage
  -- is consistent across layers.
  -- This also holds if the parameters are not learned, i.e. not adapted
  -- during training.
  self.weight = W
  self.bias = b
end

-- (3) The updateOutput method computes the forward pass of the layer.
-- The input will be the output of the previous layer and, thus,
-- may have different dimensions.
--- Compute output of layer.
-- @param intput input tensor
-- @return output
function PCA:updateOutput(input)
  if input:dim() == 2 then
    -- (3.1) Below is a simple implementation of V*(input - mean):
    local centered = input:t() - torch.repeatTensor(self.bias, input:size(1), 1):t()
    self.output = torch.mm(self.weight:t(), centered)
    self.output = self.output:t()
    return self.output
  else
    assert(false)
  end
end

-- (4) updateGradInput computes the gradients with respect to the inputs
-- by using the gradients from the top layer.
-- In this case, the backward pass is not supported, not implemented.
--- Update gradients with respect to input.
-- @param input input tensor
-- @param gradOutputs gradient outputs from top layer
-- @return gradient with respect to input
function PCA:updateGradInput(input, gradOutput)
  assert(false)
end

-- (5) As the parameters are not trainable, :parameters() is overwritten
-- to return nothing.
--- Overwrite parameters as the fixed weight and bias are not considered
-- trainable parameters.
-- See https://github.com/torch/nn/blob/master/Module.lua#L327 for an
-- error message you will get otherwise
function PCA:parameters()
  return
end

-- (6) As the parameters are not trainable, :accGradParameters
-- is overwritten to do nothing.
-- Usually, it would compute the gradients with respect to
-- self.weight and self.bias and store them in
-- self.gradWeight and self.gradBias, respectively.
--- Avoids accumulating gradient in order to fix the parameters
-- for fine-tuning.
function PCA:accGradParameters(input, gradOutput, scale)
  -- Nothing!
end