-- Some utilities.

-- https://github.com/harningt/luajson
require('json')

--- @module utils
local utils = {}

--- Print the network including all its modules.
-- @param model model to print
function utils.printModel(model)
  for i,module in ipairs(model:listModules()) do
     print(module)
  end
end

--- Checks if a file exists.
-- @see http://stackoverflow.com/questions/4990990/lua-check-if-a-file-exists
-- @param filePath path to file
-- @return true if file exists
function utils.fileExists(filePath)
  local f = io.open(filePath, 'r')
  if f ~= nill then
    io.close(f)
    return true
  else
    return false
  end
end

--- Took me 20 minutes to figure out that LUA/Torch are so f***ing stupid that this
-- is not possible without iterating!
-- @param storage storage to compute product of
-- @return product of all dimensions
function utils.storageProd(storage)
  if #storage == 0 then
    return 0
  end
  
  local prod = 1
  for i = 1, #storage do
    prod = prod * storage[i]
  end
  return prod
end

--- Compute the sum of storage elements.
-- @param storage storage to compute product of
-- @return product of all dimensions
function utils.storageSum(storage)
  local sum = 0
  for i = 1, #storage do
    sum = sum + storage[i]
  end
  return sum
end

--- Write a table as JSON to a file.
-- @param file file to write
-- @param t table to write
function utils.writeJSON(file, t)
  local f = assert(io.open(file, 'w'))
  f:write(json.encode(t))
  f:close()
end

--- Read a JSON file into a table.
-- @param file file to read
-- @return JSON string
function utils.readJSON(file)
  local f = assert(io.open(file, 'r'))
  tJSON = f:read()
  f:close()
  return json.decode(tJSON)
end

--- Recursively prints a table and all its subtables.
-- @see https://coronalabs.com/blog/2014/09/02/tutorial-printing-table-contents/
-- @param t table to print
function utils.printTable(t)
  
  -- A cache for all printed tables.
  local printCache = {}
  
  local function subPrintTable(t, indent)
    if (printCache[tostring(t)]) then
      print(indent..'*'..tostring(t))
    else
      printCache[tostring(t)]=true
      if (type(t) == 'table') then
        for pos,val in pairs(t) do
          if (type(val) == 'table') then
            print(indent..'['..pos.."] => "..tostring(t)..' {')
            subPrintTable(val,indent..string.rep(" ",string.len(pos)+8))
            print(indent..string.rep(' ',string.len(pos)+6)..'}')
          elseif (type(val) == 'string') then
            print(indent..'['..pos..'] => "'..val..'"')
          else
            print(indent..'['..pos..'] => '..tostring(val))
          end
        end
      else
        print(indent..tostring(t))
      end
    end
  end
  
  if (type(t) == 'table') then
    print(tostring(t)..' {')
    subPrintTable(t, '  ')
    print('}')
  else
    subPrintTable(t, '  ')
  end
end

--- Copies the weights from model_from to model_to for the specified layers.
-- Checks that the layers to copy weights for have the same type!
-- @param modelFrom mode to copy weights from
-- @param modelTo model to copy weights to
-- @param layersFrom layer indices in model_from
-- @param layersTo layer indices in model_to
function utils.copyWeights(modelFrom, modelTo, layersFrom, layersTo)
  assert(#layersFrom == #layersTo)
  
  for i = 1, #layersFrom do
    assert(modelTo.modules[i].__typename == modelFrom.modules[i].__typename)
    if modelTo.modules[layersTo[i]].__typename == 'nn.Linear' then
      modelTo.modules[layersTo[i]].weight = modelFrom.modules[layersFrom[i]].weight
      modelTo.modules[layersTo[i]].bias = modelFrom.modules[layersFrom[i]].bias
    end
  end
end

--- "Unsafe": Copies the weights of the given layers between two models; does not check
-- that the layers are of the same type; assumes the layers to have .weight and .bias defined.
-- @param modelFrom mode to copy weights from
-- @param modelTo model to copy weights to
-- @param layersFrom layer indices in model_from
-- @param layersTo layer indices in model_to
function utils.copyWeights(modelFrom, modelTo, layersFrom, layersTo)
  assert(#layersFrom == #layersTo)
  
  for i = 1, #layersFrom do
    modelTo.modules[layersTo[i]].weight = modelFrom.modules[layersFrom[i]].weight
    modelTo.modules[layersTo[i]].bias = modelFrom.modules[layersFrom[i]].bias
  end
end

return utils