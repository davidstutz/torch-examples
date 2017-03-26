-- Simple auto-encoder example.
-- Uses the nn.StochasticGradient trainer.

require('math')
require('torch')
require('nn')

N = 1000
D = 100

dataset = {}
function dataset:size()
  return N
end

-- (1) The dataset will contain vectors of ones and the task
-- will be to denoise these vectors; more complex datasets are easily implemented.
for i = 1, dataset:size() do 
  local output = torch.ones(D)
  local input = torch.cmul(output, torch.randn(D)*0.05 + 1)
  dataset[i] = {input, output}
end

-- (2) Setup a simple auto-encoder with bottleneck.
model = nn.Sequential()
model:add(nn.Linear(D, D/2))
model:add(nn.Tanh())
model:add(nn.Linear(D/2, D))

-- (3) Change the criterion to an absolute difference criterion.
learningRate = 0.01
criterion = nn.AbsCriterion()  
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = learningRate
trainer:train(dataset)