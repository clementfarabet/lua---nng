----------------------------------------------------------------------
--
-- Copyright (c) 2011 Clement Farabet
--
-- Permission is hereby granted, free of charge, to any person obtaining
-- a copy of this software and associated documentation files (the
-- "Software"), to deal in the Software without restriction, including
-- without limitation the rights to use, copy, modify, merge, publish,
-- distribute, sublicense, and/or sell copies of the Software, and to
-- permit persons to whom the Software is furnished to do so, subject to
-- the following conditions:
--
-- The above copyright notice and this permission notice shall be
-- included in all copies or substantial portions of the Software.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
-- NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
-- LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
-- OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
-- WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
--
----------------------------------------------------------------------
-- description:
--     nng - a neural network graph description package
--
-- history:
--     November  5, 2011, 4:57PM - first draft - Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'nn'

-- create global nng table:
nng = {}

-- import all symbols from nn
for k,v in pairs(nn) do
   nng[k] = v
end

-- create new graph
function nng.Graph(...)
   local g = {}
   g.modules = {}
   g.add = function(self, ...)
              if self.parameters then
                 error('modules cant be added to an existing graph')
              end
              local parameters = {}
              local gradParameters = {}
              for i,m in ipairs{...} do
                 table.insert(self.modules, m)
                 local p,g = m:parameters()
                 if p then
                    for _,p in ipairs(p) do
                       table.insert(parameters,p)
                    end
                    for _,g in ipairs(g) do
                       table.insert(gradParameters,g)
                    end
                 end
              end
              if parameters[1] then
                 self.parameters = nng.flatten(parameters)
                 self.gradParameters = nng.flatten(gradParameters)
              end
           end
   g.update = function(self, ...)
                 local vars = {...}
                 if #vars > 0 then
                    -- if vars given, then just update those:
                    for _,var in ipairs(vars) do
                       if var.parent then
                          var.parent:update()
                       else
                          error('the given variable has no parent in the graph')
                       end
                    end
                 else
                    -- if no var given, update the whole graph
                    for _,module in ipairs(self.modules) do
                       module:update()
                    end
                 end
              end
   local mt = {}
   mt.__tostring = function(self)
                      local str = 'nng.Graph'
                      for i,module in ipairs(self.modules) do
                         str = str .. '\n + node ' .. i .. ' of type ' .. torch.typename(module) .. ''
                         for k,input in ipairs(module.io.inputs) do
                            str = str .. '\n    - input ' .. k .. ' [valid=' .. tostring(input.valid) .. ']'
                         end
                         for k,output in ipairs(module.io.outputs) do
                            str = str .. '\n    - output ' .. k .. ' [valid=' .. tostring(output.valid) .. ']'
                         end
                      end
                      return str
                   end
   setmetatable(g,mt)
   g:add(...)
   return g
end

-- create new node
function nng.Node(module, inputs)
   -- connect inputs
   for k,input in pairs(inputs) do
      table.insert(input.children, module)
   end
   -- connect outputs
   local outputs = {nng.Var(module.output)}
   outputs[1].parent = module
   -- register all
   module.io = {}
   module.io.inputs = inputs
   module.io.outputs = outputs
   -- initially not valid
   for _,output in ipairs(module.io.outputs) do
       output.valid = false
   end
   -- update function
   module.update = function(self)
                      -- if output valid then all good
                      local isvalid = true
                      for _,output in ipairs(self.io.outputs) do
                         if not output.valid then
                            isvalid = false
                            break
                         end
                      end
                      if isvalid then
                         return
                      end
                      -- update all inputs
                      for _,input in ipairs(self.io.inputs) do
                         if not input.valid then
                            if input.parent then
                               input.parent:update()
                            else
                               error('an input of the graph is not valid, and has no parent')
                            end
                         end
                      end
                      -- update output
                      if #self.io.inputs == 1 then
                         self:forward(self.io.inputs[1].data)
                      else
                         local inputs = {}
                         for _,inp in ipairs(self.io.inputs) do
                            table.insert(inputs, inp)
                         end
                         self:forward(inputs)
                      end
                      for _,output in ipairs(self.io.outputs) do
                         output.valid = true
                      end
                   end
   return module
end

-- create new var
function nng.Var(data)
   local t = {data=data, valid=false, parent=nil, children={}}
   if data then t.valid = true end
   t.set = function(self,dest)
              self.data = dest
              self.valid = true
              for _,child in ipairs(self.children) do
                 for _,output in ipairs(child.io.outputs) do
                    output:state(false)
                 end
              end
           end
   t.state = function(self, valid)
                self.valid = valid
                for _,child in ipairs(self.children) do
                   for _,output in ipairs(child.io.outputs) do
                      output:state(valid)
                   end
                end
             end
   return t   
end

-- extend nn.Module class
local Module = torch.getmetatable('nn.Module')

function Module:__call__(inputs)
   local n = nng.Node(self, inputs)
   return n
end

-- helpers
function nng.flatten(parameters)
   -- already flat ?
   local flat = true
   for k = 2,#parameters do
      if parameters[k]:storage() ~= parameters[k-1]:storage() then
         flat = false
         break
      end
   end
   if flat then
      local nParameters = 0
      for k,param in ipairs(parameters) do
         nParameters = nParameters + param:nElement()
      end
      flatParameters = parameters[1].new(parameters[1]:storage())
      if nParameters ~= flatParameters:nElement() then
         error('weird parameters: cant deal with them')
      end
      return flatParameters
   end
   -- compute offsets of each parameter
   local offsets = {}
   local sizes = {}
   local strides = {}
   local elements = {}
   local storageOffsets = {}
   local params = {}
   local nParameters = 0
   for k,param in ipairs(parameters) do
      table.insert(offsets, nParameters+1)
      table.insert(sizes, param:size())
      table.insert(strides, param:stride())
      table.insert(elements, param:nElement())
      table.insert(storageOffsets, param:storageOffset())
      local isView = false
      for i = 1,k-1 do
         if param:storage() == parameters[i]:storage() then
            offsets[k] = offsets[i]
            if storageOffsets[k] ~= storageOffsets[i] or elements[k] ~= elements[i] then
               error('cannot flatten shared weights with different structures')
            end
            isView = true
            break
         end
      end
      if not isView then
         nParameters = nParameters + param:nElement()
      end
   end
   -- create flat vector
   local flatParameters = parameters[1].new(nParameters)
   local storage = flatParameters:storage()
   -- reallocate all parameters in flat vector
   for i = 1,#parameters do
      local data = parameters[i]:clone()
      parameters[i]:set(storage, offsets[i], elements[i]):resize(sizes[i],strides[i]):copy(data)
      data = nil
      collectgarbage()
   end
   -- cleanup
   collectgarbage()
   -- return new flat vector that contains all discrete parameters
   return flatParameters
end
