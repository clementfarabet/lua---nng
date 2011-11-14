-----------------------------------------------------------------
-- General-purpose computation, seen as a graph.  
-- Encapsulating all kinds of neural networks, and more.
--
-- Clement Farabet & Tom Schaul
--
-- TODOS:
--  + automatic unfolding in time (remove time-delays and replicate graph)
--  + correct bprop generation
--  + when cloning a node, it's akward to retrieve the input of
--    the cloned module... How do we do that ? I think that a 'group'
--    node should be aware of its inputs.
-----------------------------------------------------------------

-- dependencies
require('torch')
require('nn')

-----------------------------------------------------------------
-- Classes: Node, DataNode, TimeDelayNode
-----------------------------------------------------------------

-- Node is the fundamental object of the package. A node:
--    contains a list of dependents (children)
--    has a flag specifying whether it's OK to use it (valid)
--    may depend on other nodes (parents) for validity
local function Node(inputs)
	-- The convention we use to avoid cycles: inputs must exist upon creation
	-- and are not added later to the parents table.
	local n = {children={}, parents={}, valid=false}
	if inputs then
		for k, v in pairs(inputs) do 
			n.parents[k] = v
			table.insert(v.children, n) 
		end
	end
	
	-- Core method: validates all inputs and then produces a valid output 
	function n.read(...)
		-- recursive propagation to parents		
		for _, p in pairs(n.parents) do
			if not p.valid then p.read() end
		end
		n.valid = true
		-- the return value is either coming from a function call, or 
		-- can return a pointer to some data
		if n.guts and type(n.guts)=='function' then 
			return n.guts(...)
		else
			return n.guts
		end		
	end
	
	-- Its partner method invalidates all dependants (maybe with a side-effect)
	function n.write()
		n.valid = false		
		for _,n in pairs(n.children) do
			if n.valid then n.write() end
		end
	end
		
	-- A synchronized tick signal (default: no effect)	
	function n.tick() end
	
    local mt = {}
    function mt.__tostring(self)
		local s = n.name or ''
		return s
	end
	setmetatable(n,mt)	
	return n
end

-- Data nodes depend only on their data, which is valid as long as it has not 
-- been overwritten. Note that data can just as well be a closure that generates
-- or reads something when invoked
-- TODO: should those nodes know about their size, even before they are valid? 
--       To help automatic constructions avoid runtime errors? 
local function DataNode(data)
	local n = Node()
	n.name = "Data"
	n.guts = data
	n.valid = (data ~= nil) 
	-- update the data, and propagate the invalid flag to all children
	function n.write(newdata)
		if newdata then n.guts = newdata end
		n.valid = (newdata ~= nil) 
		for _,c in pairs(n.children) do 
			if c.valid then c.write() end
		end				
	end
	return n
end

-- A time-delayed Node permits safely introducing cyles, so it does not need 
-- to know its inputs upon construction
local function TimeDelayNode(size, initvalues)
	local t = Node()
	t.name = "TimeDelay"
	if not initvalues then initvalues = lab.zeros(size) end
	t.output = DataNode(initvalues)
	t.children = {t.output}
	
	-- The node must be connceted to a single datanode input
	-- (tensor of the pre-specified size)
	function t.connectInput(input)
		t.input = input
		t.valid = true
	end
	
	-- At each time-stpe, copy the input data from the past.
	function t.tick()
		t.output.write(torch.Tensor(size):copy(t.input.read()))					
	end
	return t
end

-----------------------------------------------------------------
-- Extend nn modules to act as Nodes
-----------------------------------------------------------------

-- Helper function to make single inputs and tables more 
-- transparent (private func)
local function nodetable2inputs(nodetable)
	if #nodetable == 1 then
		return nodetable[1].read()
	else
		local t = {}
		for i,input in ipairs(nodetable) do 
			t[i] = input.read()
		end
		return t
	end
end

-- This extends each nn.Module class, such that its call operator wraps it in a node
local Module = torch.getmetatable('nn.Module')
function Module:__call__(inputs)
	local n = Node(inputs)
	n.name = torch.typename(self)
	n.module = self
	n.inputs = inputs
	-- Creating wrappers around anything that can change, 
	n.output = DataNode(self.output)
	n.output.valid = false
	local p,_ = self:parameters()
	n.parameters = DataNode(p)
	-- and establish the dependencies
	table.insert(n.parents, n.parameters)
	n.children = {n.output}
	n.output.parents={n}
	
	function n.guts()
		n.module:forward(nodetable2inputs(n.inputs))	
		n.output.write(n.module.output)
	end
		
	return n
end

-- This extends each nn.Criterion class, such that its call operator wraps it in a node
local Criterion = torch.getmetatable('nn.Criterion')
function Criterion:__call__(inputs)
	local n = Node(inputs)
	n.name = torch.typename(self)
	n.module = self
	n.inputs = inputs
	-- Creating wrappers around anything that can change, 
	n.output = DataNode(self.output)
	n.output.valid = false
	-- and establish the dependencies
	n.children = {n.output}
	n.output.parents={n}
	
	function n.guts()
		local ins = nodetable2inputs(n.inputs)
		-- a criterion always has two inputs
		n.module:forward(ins[1], ins[2])
		n.output.write(n.module.output)
	end
		
	return n
end

-----------------------------------------------------------------
-- Package functions
-----------------------------------------------------------------

-- Nodes can be grouped together, in a new node (deep nesting is fine).
--     default: a single output
local function groupNodes(nodes, output)
	local gn = Node()
	gn.nodes = nodes
	gn.output = output
	
	-- TODO: something fancier..
	local s = "Group["
	for _, n in pairs(nodes) do 
		if n.name then s = s.." "..n.name end 	
	end
	gn.name = s.." ]" 
	
	-- Ticks are propagated to all members
	function gn.tick()
		for _, n in pairs(gn.nodes) do
			if n.tick then n.tick() end
		end				
	end
	return gn
end


-- Builds the twin node for the backward pass
local function buildBackwardNode(node, gradOutputs)
	assert (#gradOutputs > 0)
	assert (node.module)
	
	local twin = Node(gradOutputs)
	table.insert(twin.parents, node)
	table.insert(node.children, twin)
	twin.name = node.name.."twin"
	
	-- One or two datanodes depend on the twin 
	twin.output = DataNode(node.module.gradInput)
	twin.output.valid=false
	table.insert(twin.output.parents, twin)
	table.insert(twin.children, twin.output)
	local _,gp = node.module:parameters()	
	if gp then
		twin.gradParameters = DataNode(gp)
		twin.gradParameters.valid=false
		table.insert(twin.gradParameters.parents, twin)
		table.insert(twin.children, twin.gradParameters)
	end
	twin.gradOutputs = gradOutputs
	
	function twin.guts()
		-- The convention is that all gradoutputs can be summed
		local gradOutput = twin.gradOutputs[1].read()
		for i=2,#twin.gradOutputs do
			gradOutput = gradOutput + twin.gradOutputs[i].read()			
		end		
		twin.output.write(node.module:backward(nodetable2inputs(node.inputs), gradOutput))
	end
	node.twin = twin	
	return twin
end


-- Recursively builds the twin/backward nodes for a network (the starting point 
--     is a node whose output corresponds to the provided external error)
--     Returns a list of all twins created,
--     or 'nil' if a dependency was missing.
local function backwardTwin(node, extGradOutput, reccall)
	local result = {}
	
	-- if the node already has a twin, we're done
	if node.twin then return {} end
		
	-- nesting case (retains the nesting structure for the backward pass)
	if node.nodes then
		result = backwardTwin(node.output, extGradOutput, reccall)
		if result then
			local mine = {}		
			for i,n in ipairs(result) do
				if n.module and not n.twin then return nil end
				for _,nn in node.nodes do
					if n==nn then
						table.insert(mine, n.twin)
						table.remove(result, i)
						break
					end
				end
			end
			node.twin = groupNodes(mine)
			table.insert(result, node.twin)
		end
		return result
	end 
	
	-- standard case (node with a module)
	if node.module then
		-- the outputs used come from the children's twins (plus the external one) 
		local gradOutputs = {}
		if extGradOutput then 
			gradOutputs = {extGradOutput} 
		end
	
		-- we make sure all the node's children are usable
		-- and if they are, we will use their twins
		local function getTwins(cn)
			if cn.twin then
				table.insert(gradOutputs, cn.twin.output)
			else
				for _,cc in ipairs(cn.children) do
					getTwins(cc)
				end
			end
		end
		for _,cn in ipairs(node.children) do
			local tmp = backwardTwin(cn, nil, true)
			if not tmp then return nil end
		 	getTwins(cn)
		end
					
		-- build a twin for this node
		buildBackwardNode(node, gradOutputs)
		result = {node.twin}
	end
	
	
	-- Now that this node has a twin, its parents can get them too (recursively)	
	if reccall then return result end
	for _,p in ipairs(node.parents) do
		if node.twin then
			local tmp = backwardTwin(p)
		else
			-- the external info was not used, so it's passed up one level
			local tmp = backwardTwin(p, extGradOutput)		
		end
		if tmp then
			for _,x in ipairs(tmp) do
				table.insert(result, x)
			end
		end		
	end
	return result

end


-- Flatten helper (private function)
local function flattenParameters(parameters)
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

-- Parameter finder
local function getParameters(nodes, params)
	local params = params or {}
	for _,node in pairs(nodes) do
		if node.parameters and node.parameters.guts then
			for _,p in pairs(node.parameters.guts) do
				table.insert(params, p)
			end
		end
		if node.nodes then
			getParameters(node.nodes, params)
		end
	end
	return params
end

-- Share parameters: takes a list of nodes, and share parameters
-- between these nodes. They should have the same structure of course.
local function shareParameters(nodes)
	local params = {}
	for _,n in pairs(nodes) do
		table.insert(params, getParameters{n})
	end
	for i = 2,#params do
		local ref = params[1]
		local param = params[i]
		for k = 1,#param do
			param[k]:set(ref[k])
		end
	end
end

-- Clone node
local function cloneNode(node, share)
	local f = torch.MemoryFile()
	f:writeObject(node)
	f:seek(1)
	local cloned = f:readObject()
	f:close()
	if share then
		shareParameters{node,cloned}
	end
	return cloned
end

-- Inspects all the nodes for data marked with a "parameter" flag
-- and flattens their storage (as well as, symetrically, the storage
-- of the corresponding derivatives).
local function flattenNodes(nodes)
	local params = getParameters(nodes)
	local flat = flattenParameters(params)
	return flat
end

-- register functions in package
g = {
	Node = Node, 
	DataNode = DataNode, 
	groupNodes = groupNodes,
	cloneNode = cloneNode,
	backwardTwin = backwardTwin,
	flattenNodes = flattenNodes,
	getParameters = getParameters,
	shareParameters = shareParameters,
	TimeDelayNode = TimeDelayNode
}

-----------------------------------------------------------------
-- A few examples of composite nodes that can be built
-----------------------------------------------------------------

-- A recurrent counter	
function g.CounterNode()
	-- the flipflop is built first
	local fflop = TimeDelayNode(1)
	-- the linear transformation does: x <- 1*x+1 
	local mod = nn.Linear(1,1){fflop.output}	
	-- TODO: this is not best way:
	mod.module.weight:fill(1) 
	mod.module.bias:fill(1)	
	-- the flipflop is connected at the end
	fflop.connectInput(mod.output)	
	return groupNodes({fflop, mod}, mod.output)
end

-- A general-purpose MLP constructor 
function g.MultiLayerPerceptron(sizes, input)
	local layers = {}
	local last = input
	for i=2,#sizes do
		local affine = nn.Linear(sizes[i-1], sizes[i]){last}
		local squash = nn.Tanh(){affine.output}
		last = squash.output
		table.insert(layers, affine)
		table.insert(layers, squash)
	end
	return groupNodes(layers, last)
end

-- A standard ConvNet
function g.ConvNet(nfeatures, fanins, filters, poolings, input)
	local layers = {}
	local last = input
	for i=2,#nfeatures do
		local c = nn.SpatialConvolution(nfeatures[i-1], nfeatures[i], filters[i-1], filters[i-1]){last}
		table.insert(layers, c)
		if i ~= #nfeatures then
			local s = nn.Tanh(){c.output}
			local p = nn.SpatialMaxPooling(poolings[i-1], poolings[i-1]){s.output}
			table.insert(layers, s)
			table.insert(layers, p)
			last = p.output
		else
			last = c.output
		end
	end
	return groupNodes(layers, last)
end

-- An Elman network has three fully connected layers (in, hidden, out),
-- with the activations of the hidden layer feeding back into the 
-- input, with a time-delay.
function g.ElmanNode(sizes, input)
	assert (#sizes == 3)
	local fflop = TimeDelayNode(sizes[2])
	local mod0 = nn.JoinTable(1){input, fflop.output}
	local mod1 = nn.Linear(sizes[1]+sizes[2],sizes[2]){mod0.output}
	local mod2 = nn.Tanh(){mod1.output}
	local mod3 = nn.Linear(sizes[2],sizes[3]){mod2.output}
	-- connecting recurrent link
	fflop.connectInput(mod2.output)
	return groupNodes({fflop, mod0, mod1, mod2, mod3}, mod3.output)	
end

-- Long short-term memory cells (LSTM) are a specialized building block
-- of recurrent networks, useful whenever long time-lags need to be captured
-- (information conserved over a prolonged time in the activations).
-- takes 4 input Var objects, returns the required modules and an output Var
-- TODO: add peephole connections
function g.LstmUnit(size, datain, ingatein, forgetgatein, outgatein)
	-- all the gate inputs get squashed between [0,1]
	local ingate = nn.Sigmoid(){ingatein}
	local forgetgate = nn.Sigmoid(){forgetgatein}
	local outgate = nn.Sigmoid(){outgatein}
	-- data is squashed too, then gated
	local newdata = nn.Tanh(){datain}
	local statein = nn.CMulTable(){ingate.output, newdata.output}
	-- the inner "carousel" retains the state information indefinitely
	-- as long as the forgetgate is not used (gated data is added)
	local fflop = TimeDelayNode(size)
	local state = nn.CAddTable(){statein.output, fflop.output}
	local nextstate = nn.CMulTable(){forgetgate.output, state.output}
	fflop.connectInput(nextstate.output)
	-- one last squashing, of the output
	local preout = nn.CMulTable(){outgate.output, state.output}
	local out = nn.Tanh(){preout.output}
	
	return groupNodes({ingate, forgetgate, outgate, newdata, statein, 
					   fflop, state, nextstate, preout, out}, out.output)
end

-- return package
return g
