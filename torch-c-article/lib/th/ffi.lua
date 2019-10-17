-- Include C modules.

require('os')
local ffi = require('ffi')

-- Will contain all C modules later ...
lib.cpu = {}
lib.gpu = {}

ffi.cdef[[
void test_module_updateOutput(const int rank, const long* dims, const float* input, float* output);
void test_module_updateGradInput(const int rank, const long* dims, const float* input, const float* grad_output, float* grad_input);
void test_identity_module_updateOutput(const int rank, const long* dims, const float* input, float* output);
void test_identity_module_updateGradInput(const int rank, const long* dims, const float* input, const float* grad_output, float* grad_input);
]]

--- Get the script path;
-- this assumes that the LUA modules are contained in ../th/.
-- @return script path
local function scriptPath()
  local str = debug.getinfo(2, "S").source:sub(2)
  return str:match("(.*/)")
end

-- The path to the shared library.
local libname = scriptPath() .. '../cpp/cpu/build/libcpu.so'
local found = pcall(function () lib.cpu = ffi.load(libname) end)

if found then
  print('[Lib] found ' .. libname)
else
  print('[Info] could not find CPU module, tried ' .. libname)
  print('[Info] will continue without CPU module')
  lib.cpu = false
  os.exit()
end

-- Same as for CPU above.
if cutorch then
  ffi.cdef[[
  void test_identity_module_updateOutput(const int rank, const long* dims, const float* input, float* output);
  void test_identity_module_updateGradInput(const int rank, const long* dims, const float* input, const float* grad_output, float* grad_input);
  ]]

  local libname = scriptPath() .. '../cpp/gpu/build/libgpu.so'
  local found = pcall(function () lib.gpu = ffi.load(libname) end)

  if found then
    print('[Lib] found ' .. libname)
  else
    print('[Info] could not find GPU module, tried ' .. libname)
    print('[Info] will continue without GPU module')
    lib.gpu = false
    os.exit()
  end
end