-- Small example to test forward and packward passes of custom data structures.

require('math')
require('torch')
require('nn')
require('lfs')

package.path = package.path .. ";" .. lfs.currentdir() .. '?.lua'
init = require('init')

--- @class CustomDataStructure
-- This calss will be out simple test data structure.
CustomDataStructure = {}
CustomDataStructure.__index = CustomDataStructure

--- Creates a new CustomDataStructure with 0-vectors of the given size
-- @param n vector size
function CustomDataStructure.create(n)
  local cds = {}
  setmetatable(cds, CustomDataStructure)
  cds.x1 = torch:Tensor(n):fill(0)
  cds.x2 = torch:Tensor(n):fill(0)
  return cds
end

--- @class CustomLinear
CustomLinear, CustomLinearParent = torch.class('nn.CustomLinear', 'nn.Module')

--- Initialize the layer specifying the number of input and output units.
-- @param nInputUnits number of input units
-- @param nOutputUnits number of output units
function CustomLinear:__init__(nInputUnits, nOutputUnits)
  self.nInputUnits = nInputUnits
  self.nOutputUnits = nOutputUnits
  self.weight = torch.Tensor(nOutputUnits, nInputUnits):fill(0)
end

--- Compute output.
-- @param input input of type CustomDataStructure
-- @return output of type CustomDataStructure
function CustomLinear:updateOutput(input)
  self.output = CustomDataStructure.create(self.nOutputUnits)
  self.output.x1 = torch.mv(self.weight, input.x1)
  self.output.x2 = torch.mv(self.weight, input.x2)
  return self.output
end

--- Avoid backward pass.
function CustomLinear:UpdateGradInput(input, gradOutput)
  assert(false)
end

N = 10
x1 = torch.Tensor(N):fill(0)
x2 = torch.Tensor(N):fill(1)
x = CustomDataStructure.create(N)
x.x1 = x1
x.x2 = x2

model = nn.Sequential()
module = nn.CustomLinear(10, 1)
module.weight = torch.Tensor(1, 10):fill(1)
model:add(module)

y = model:forward(x)
print(y.x1)
print(y.x2)
