-- Include CPU/GPU modules first.

lib = {}
include('ffi.lua')
include('TestModule.lua')
include('TestIdentityModule.lua')

return lib