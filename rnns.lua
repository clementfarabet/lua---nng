-- Recurrent networks: flip-flop units and some common topologies
-- (Tom Schaul)

require 'nng'
require 'lab'


-- TODO: longer delays than 1
function nng.FlipFlop(size, initvalues)
	if not initvalues then initvalues = lab.zeros(size) end
	local t = nng.Identity()
	t.io = {outputs={nng.Var(initvalues), parent=t}}
	t.connect = function (self, inputs)
					assert(#inputs == 1)
				   	--table.insert(inputs[1].children, t)
				   	t.io.inputs = inputs   					
				end
	-- dummy functions: flipflops don't propagate this
	t.io.outputs.state = function (self, valid) end
	t.forward = function (self, input) end
	t.update = function (self, input) end 
	-- key function
	t.tick = 	function (self)
					if not self.io.inputs[1].valid then
						self.io.inputs[1].parent:update()
					end
					self.io.outputs[1]:set(torch.Tensor(size):copy(self.io.inputs[1].data))					
				end
	return t
end


-- An Elman network has three fully connected layers (in, hidden, out),
-- with the activations of the hidden layer feeding back into the 
-- input, with a time-delay.
function elmanNodes(sizes)
	assert (#sizes == 3)
	in1 = nng.Var()
	fflop = nng.FlipFlop(sizes[2])
	m0 = nng.JoinTable(1){in1, fflop.io.outputs[1]}	
	m1 = nng.Linear(sizes[1]+sizes[2],sizes[2])(m0.io.outputs)
	--m1 = nng.Linear(sizes[1],sizes[2]){invar}
	m2 = nng.Tanh()(m1.io.outputs)
	fflop:connect(m2.io.outputs) -- connecting recurrent link
	--m0:update()
	print(m0.io.inputs[1], m0.io.inputs[2], m0.io.outputs[1])
	m3 = nng.Linear(sizes[2],sizes[3])(m2.io.outputs)
	return {fflop, m0, m1, m2, m3}, in1, m3.io.outputs[1]	
end


-- let's build a recurrent counter	
function testCounter()
	-- the flipflop is built first
	ff = nng.FlipFlop(1)
	-- the linear transformation does: x <- 1*x+1 
	mod = nng.Linear(1,1){ff.io.outputs[1]}	
	mod.weight:fill(1) 
	mod.bias:fill(1)	
	-- the flipflop is connected at the end
	ff:connect(mod.io.outputs)	
	g = nng.Graph(ff, mod)
	
	output = mod.io.outputs[1]
	print(g, "init")
	for i=1,10 do
		g:update(output)
		print(output.data[1])
		g:tick()		
	end
	g:update(output)		
	print(g, "end")	
end


-- a simple neural network that produces the square roots of the fibonacci numbers	
function testFibonacci()
	-- the flipflop is built initialized with (0,1)
	init = torch.Tensor(2):zero()
	init[2] = 1
	ff = nng.FlipFlop(2, init)
	-- the linear transformation does: x,y <- y, x+y 
	mod = nng.Linear(2,2){ff.io.outputs[1]}	
	mod.weight:fill(1)
	mod.weight[1][1]=0 
	mod.bias:fill(0)	
	ff:connect(mod.io.outputs)
	omod = nng.Sqrt()(mod.io.outputs)	
	g = nng.Graph(mod, omod, ff)
	
	output = omod.io.outputs[1]
	-- let's see if the 12th member of the sequence is indeed 12^2...
	for i=1,12 do
		g:update(output)
		print(i, output.data[1])
		g:tick()
	end
end


-- basic testing of the network flow in an elman network
function testElman()
	modules, invar, out = elmanNodes{1,2,3}
	g = nng.Graph(unpack(modules))
	print(g, "init")
	invar:set(lab.zeros(1))
	print(g, "set", out.data)
	g:update(out)
	print(g, "updated", out.data)
	g:tick()
	print(g, "ticked", out.data)
	g:update(out)
	-- note that the input has not changed in the meanwhile
	print(g, "updated again", out.data)	
end


--testCounter()
--testFibonacci()
--testElman()

