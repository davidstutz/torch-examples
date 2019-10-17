require('torch')
require('nn')
require('optim')
require('hdf5')
require('cunn')
require('lfs')

package.path = package.path .. ";" .. lfs.currentdir() .. '/?/th/init.lua'
lib = require('lib')

model = nn.Sequential()
model:add(nn.TestIdentityModule())

model:forward(torch.randn(10, 10):float())
model:forward(torch.randn(10, 10):cuda())