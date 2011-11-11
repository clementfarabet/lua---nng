----------------------------------------------------------
-- General-purpose computation, seen as a graph.  
-- Encapsulating all kinds of neural networks, and more.
--
-- Clement Farabet & Tom Schaul
----------------------------------------------------------


-- TODO: copying methods (with/without weight sharing)
-- TODO: automatic unfolding in time


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
function DataNode(data)
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



-- Nodes can be grouped together, in a new node (deep nesting is fine).
--     default: a single output
function groupNodes(nodes, output)
	local g = Node()
	g.nodes = nodes
	g.output = output
	
	-- TODO: something fancier..
	s = "Group["
	for _, n in pairs(nodes) do 
		if n.name then s = s.." "..n.name end 	
	end
	g.name = s.." ]" 
	
	-- Ticks are propagated to all members
	function g.tick()
		for _, n in pairs(g.nodes) do
			if n.tick then n.tick() end
		end				
	end
	return g
end


-- This extends each nn.Module class, such that its call operator wraps it in a node
require('nn')
require('torch')
-- TODO: package management not finished
nngg = {}
-- import all symbols from nn
for k,v in pairs(nn) do
   nngg[k] = v
end

local Module = torch.getmetatable('nn.Module')

function Module:__call__(inputs)	
	local n = Node(inputs)
	n.name = torch.typename(module)
	-- Creating wrappers around anything that can change, 
	n.module = self
	n.inputs = inputs
	n.output = DataNode(self.output)
	n.output.valid = false
	n.output.parents={n}
	n.parameters = DataNode(self.parameters)
	n.gradParameters = DataNode(self.gradParameters)
	-- and establish the dependencies
	table.insert(n.parents, n.parameters)
	table.insert(n.parents, n.gradParameters)
	n.children = {n.output}
	
	function n.guts()
		if #n.inputs == 1 then
			n.module:forward(n.inputs[1].read())
		else
			local t = {}
			for i,input in ipairs(n.inputs) do 
				t[i] = input.read()
			end
			n.module:forward(t)
		end
		n.output.write(n.module.output)
	end
	
	-- TODO: create a node for the backward pass now too!
	
	return n
end


-- Inspects all the nodes for data marked with a "parameter" flag
-- and flattens their storage (as well as, symetrically, the storage
-- of the corresponding derivatives).
function flattenNodes(nodes)
	--- TODO
end


-- A time-delayed Node permits safely introducing cyles, so it does not need 
-- to know its inputs upon construction
function TimeDelayNode(size, initvalues)
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
-- A few examples of composite nodes that can be built
-----------------------------------------------------------------


-- A recurrent counter	
function CounterNode()
	-- the flipflop is built first
	fflop = TimeDelayNode(1)
	-- the linear transformation does: x <- 1*x+1 
	mod = nngg.Linear(1,1){fflop.output}	
	-- TODO: this is not best way:
	mod.module.weight:fill(1) 
	mod.module.bias:fill(1)	
	-- the flipflop is connected at the end
	fflop.connectInput(mod.output)	
	return groupNodes({fflop, mod}, mod.output)
end


-- An Elman network has three fully connected layers (in, hidden, out),
-- with the activations of the hidden layer feeding back into the 
-- input, with a time-delay.
function elmanNode(sizes, input)
	assert (#sizes == 3)
	local fflop = TimeDelayNode(sizes[2])
	local mod0 = nngg.JoinTable(1){input, fflop.output}
	local mod1 = nngg.Linear(sizes[1]+sizes[2],sizes[2]){mod0.output}
	local mod2 = nngg.Tanh(){mod1.output}
	local mod3 = nngg.Linear(sizes[2],sizes[3]){mod2.output}
	-- connecting recurrent link
	fflop.connectInput(mod2.output)
	return groupNodes({fflop, mod0, mod1, mod2, mod3}, mod3.output)	
end


-- Long short-term memory cells (LSTM) are a specialized building block
-- of recurrent networks, useful whenever long time-lags need to be captured
-- (information conserved over a prolonged time in the activations).
-- takes 4 input Var objects, returns the required modules and an output Var
-- TODO: add peephole connections
function lstmUnit(size, datain, ingatein, forgetgatein, outgatein)
	-- all the gate inputs get squashed between [0,1]
	local ingate = nngg.Sigmoid(){ingatein}
	local forgetgate = nngg.Sigmoid(){forgetgatein}
	local outgate = nngg.Sigmoid(){outgatein}
	-- data is squashed too, then gated
	local newdata = nngg.Tanh(){datain}
	local statein = nngg.CMulTable(){ingate.output, newdata.output}
	-- the inner "carousel" retains the state information indefinitely
	-- as long as the forgetgate is not used (gated data is added)
	local fflop = TimeDelayNode(size)
	local state = nngg.CAddTable(){statein.output, fflop.output}
	local nextstate = nngg.CMulTable(){forgetgate.output, state.output}
	fflop.connectInput(nextstate.output)
	-- one last squashing, of the output
	local preout = nngg.CMulTable(){outgate.output, state.output}
	local out = nngg.Tanh(){preout.output}
	
	return groupNodes({ingate, forgetgate, outgate, newdata, statein, 
					   fflop, state, nextstate, preout, out}, out.output)
end


-----------------------------------------------------------------
-- A few basic tests cases (TODO: expand)
-----------------------------------------------------------------


-- Check how are asynchronous updates and queries handled.
function testValidityPropagation()
	local input1 = DataNode()
	local input2 = DataNode()
	local mod1 = nngg.Linear(10,10){input1}
	local mod2 = nngg.Linear(10,10){input2}
	local mod3 = nngg.JoinTable(1){mod1.output, mod2.output}		
	local mod4 = nngg.Tanh(){mod3.output}
	local output2 = mod2.output
	local output4 = mod4.output
	local function g1()
		local l = {'i',input1.valid,input2.valid,
				   '-m', mod1.valid,mod2.valid,mod3.valid,mod4.valid,
				   '-o',output2.valid,output4.valid}
		local s = ''
		for _,v in ipairs(l) do
			if v ==true then
				s = s.."1"
			elseif v==false then
				s = s.."0"
			else
				s = s..v
			end
		end
		return s
	end
	print(g1(), "invalid inputs, invalid outputs") --OK
	input1.write(lab.randn(10))
	print(g1(), "valid input1, invalid input2, invalid outputs") --OK
	input2.write(lab.randn(10))
	print(g1(), "valid inputs, invalid outputs") --OK
	output2.read()
	print(g1(), "valid output2, invalid output3") --OK
	output4.read()
	print(g1(), "valid outputs") --OK
	input1.write(lab.randn(10))
	print(g1(), "valid output 2, invalid output3") --OK
	input2.write(lab.randn(10))
	print(g1(), "invalid outputs") --OK
	output4.read()
	print(g1(), "valid outputs") --OK
end


-- Nesting can of groups of nodes can be arbitraily deep
-- This example recursively builds a deep MLP like that
-- TODO: also test nested flattening
function testNesting()
	local depth = 100
	local size = 5
	local base = DataNode()
	local top = groupNodes({}, base)
	for _=1,depth do
		local affine = nngg.Linear(size, size){top.output}
		local squash = nngg.Tanh(){affine.output}
		top = groupNodes({top, affine, squash}, squash.output)		
	end
	base.write(lab.ones(size))
	print(top.output.read())
end


function testCounter()
	local cnode = CounterNode()
	-- without ticks nothing budges
	for i=1,3 do
		print(cnode.output.read()[1].."="..1)
	end
	-- with ticks it counts
	for i=2,6 do
		cnode.tick()	
		print(cnode.output.read()[1].."="..i, "ticked!")			
	end	
end


-- Basic testing of the network flow in an Elman network
-- note that once set, the input to the net does not change here
local function testElman()
	local input1 = DataNode()
	local en = elmanNode({3,9,2}, input1)
	input1.write(lab.zeros(3))
	print(en.output.read()[1], "first pass")
	print(en.output.read()[1], "first pass (still)")
	en.tick()
	print(en.output.read()[1], "second pass")
	en.tick()
	print(en.output.read()[1], "third pass")		
end


-- A simple neural network that produces the square roots of the Fibonacci numbers	
function testFibonacci()
	-- the flipflop is initialized with (0,1)
	local init = torch.Tensor(2):zero()
	init[2] = 1
	local ff = TimeDelayNode(2, init)
	-- the linear transformation does: x,y <- y, x+y 
	local mod = nngg.Linear(2,2){ff.output}
	mod.module.weight:fill(1)
	mod.module.weight[1][1]=0 
	mod.module.bias:fill(0)	
	ff.connectInput(mod.output)
	local omod = nngg.Sqrt(){mod.output}	
	local fibnode = groupNodes({mod, omod, ff}, omod.output)
	
	-- let's see if the 12th member of the sequence is indeed 12^2...
	for i=1,12 do
		print(i, fibnode.output.read()[1])
		fibnode.tick()
	end
end


-- Illustrate the LSTM gating
function testLSTM()
	local size=2
	local datain = DataNode()
	local ingatein = DataNode()
	local forgetgatein = DataNode()
	local outgatein = DataNode()
	local lstmnode = lstmUnit(size, datain, ingatein, forgetgatein, outgatein)
	local outvar = lstmnode.output 
	
	-- input data: [0.01, 0.1] 
	local incs = lab.ones(size)*0.1
	incs[1]= 0.02
	datain.write(incs)
	
	-- gates completely open
	local open = lab.ones(size)*1000
	ingatein.write(open) 
	forgetgatein.write(open) 
	outgatein.write(open)
	
	for i=1,10 do
		print(i, outvar.read()[1], outvar.read()[2])
		lstmnode.tick()
	end	
	print()
	
	-- close input gate on one unit 
	local halfopen = lab.ones(size)*1000
	halfopen[1] = -1000
	ingatein.write(halfopen) 
	for i=1,5 do
		print(i, outvar.read()[1], outvar.read()[2])
		lstmnode.tick()
	end
	print()
	
	-- forget immediately on the other one now 
	local closed = lab.ones(size)*-1000
	forgetgatein.write(closed) 
	for i=1,5 do
		print(i, outvar.read()[1], outvar.read()[2])
		lstmnode.tick()
	end
	
end


-- run all the tests
testCounter()
testValidityPropagation()
testElman()
testFibonacci()
testLSTM()
testNesting()
