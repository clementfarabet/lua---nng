----------------------------------------------------------
-- General-purpose computation, seen as a graph.  
-- Encapsulating all kinds of neural networks, and more.
--
-- Clement Farabet & Tom Schaul
----------------------------------------------------------



-- Node is the fundamental object of the package. A node:
--    contains a list of dependents (children)
--    has a flag specifying whether it's OK to use it (valid)
--    may depend on other nodes (parents) for validity
local function Node(inputs)
	-- The convention we use to avoid cycles: inputs must exist upon creation
	-- and are not added later to the parents table.
	local n = {children={}, parents=inputs, valid=false}
	
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


-- Nodes can be grouped together, in a new node.
function groupNodes(nodes)
	local g = Node()
	g.nodes = nodes -- XX
	
	s = "Group["
	for _, n in nodes do 
		if n.name then s = s.." "..n.name end 	
	end
	g.name = s.."]" 
	
	-- Ticks are propagated to all members
	function g.tick()
		for _, n in pairs(g.nodes) do
			if n.tick then n.tick() end
		end				
	end
	return g
end


-- Data nodes depend only on their data, which is valid as long as it has not 
-- been overwritten. Note that data can just as well be a closure that generates
-- or reads something when invoked
function DataNode(data)
	local n = Node()
	n.name = "Data"
	n.guts = data
	-- update the data, and propagate the invalid flag to all children
	function n.write(newdata)
		n.guts = newdata
		n.valid = false		
	end
	return n
end


-- A time-delayed Node permits safely introducing cyles, so it does not need 
-- to know its inputs upon construction
function TimeDelayNode(size, initvalues)
	local t = Node()
	t.name = "TimeDelay"
	if not initvalues then initvalues = lab.zeros(size) end
	t.children = {DataNode(initvalues)}
	
	-- The node must be connceted to a single datanode input
	-- (tensor of the pre-specified size)
	function t.connect(input)
		t.input = input
		t.valid = true
	end
	
	-- At each time-stpe, copy the input data from the past.
	function t.tick()
		t.children[1].write(torch.Tensor(size):copy(t.input.read()))					
	end
	return t
end


-- This extends each nn.Module class, such that its call operator wraps it in a node
require('torch')
local Module = torch.getmetatable('nn.Module')

function Module:__call__(inputs)
	
	-- XX
	local n = Node(inputs)
	n.module = self
	-- n.children 
	-- XX
	
	function n.guts()
		n.module.forward(n.parents)
	end
	-- XX create a node for the backward pass now too!
end



-- Inspects all the nodes for data marked with a "parameter" flag
-- and flattens their storage (as well as, symetrically, the storage
-- of the corresponding derivatives).
function flattenNodes(nodes)
	
end


-- XX Fancy syntax?
function decorateNode(n)
    local mt = {}
    function mt.__index(self, ...) 
    	return self.read(...) 
    end
	function mt.__newindex(self, ...) 
		self.write(...) 
	end
	setmetatable(n,mt)	
end




-- Tests go here... 


