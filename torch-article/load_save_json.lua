require('json')
require('torch')

--- Write a table as JSON to a file.
-- @param file file to write
-- @param t table to write
function writeJSON(file, t)
  local f = assert(io.open(file, 'w'))
  f:write(json.encode(t))
  f:close()
end

--- Read a JSON file into a table.
-- @param file file to read
-- @return JSON string
function readJSON(file)
  local f = assert(io.open(file, 'r'))
  tJSON = f:read()
  f:close()
  return json.decode(tJSON)
end

--- Recursively prints a table and all its subtables.
-- @see https://coronalabs.com/blog/2014/09/02/tutorial-printing-table-contents/
-- @param t table to print
function printTable(t)
  
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

t = {}
t['test'] = 10
t[1] = 11
t[2] = 'test'
t[3] = torch.rand(5, 5):totable()

writeJSON('test.json', t)
printTable(readJSON('test.json'), 'ttt')