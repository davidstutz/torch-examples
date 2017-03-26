-- Simple 2D classification example taken somewhere from the Torch docs.
-- Uses the nn.StochasticGradient trainer.

require('torch')
require('nn')

-- (1) Setup the dataset, nn.StochasticGradient requires a :size() method.
dataset = {}
function dataset:size()
  return 100
end

-- (2) The dataset should contain inputs and outputs (targets) such that dataset[i][0] 
-- is the input of sample i and dataset[i][0] the corresponding outputs.
for i = 1, dataset:size() do 
  local input = torch.randn(2)
  local output = torch.Tensor(1)
  if input[1]*input[2] > 0 then
    output[1] = -1;
  else
    output[1] = 1
  end
  dataset[i] = {input, output}
end

-- (3) Setup a simple classifier by adding the corresponding layers
-- to a nn.Sequential container.
model = nn.Sequential()
model:add(nn.Linear(2, 20))
model:add(nn.Tanh())
model:add(nn.Linear(20, 1))

-- (4) nn.StochasticTrainer expects a model and a criterion (here the MSECriterion)
-- for training.
learningRate = 0.01
criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = learningRate
trainer:train(dataset)