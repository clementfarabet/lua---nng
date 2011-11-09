-- Recurrent networks: flip-flop units and some common topologies
-- (Tom Schaul)

require 'nng'
require 'lab'


-- TODO: longer delays than 1
function nng.FlipFlop(size, initvalues)
	if not initvalues then initvalues = lab.zeros(size) end
	local t = nng.Identity()
	t.io = {outputs={nng.Var(initvalues), parent=t}}
	-- finishes up the flipflop construction, called at a later point, 
	-- when the inputs are available. 
	t.connect = function (self, inputs)
					assert(#inputs == 1)
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
	m2 = nng.Tanh()(m1.io.outputs)
	fflop:connect(m2.io.outputs) -- connecting recurrent link
	print(m0.io.inputs[1], m0.io.inputs[2], m0.io.outputs[1])
	m3 = nng.Linear(sizes[2],sizes[3])(m2.io.outputs)
	return {fflop, m0, m1, m2, m3}, in1, m3.io.outputs[1]	
end

-- Long short-term memory cells (LSTM) are a specialized building block
-- of recurrent networks, useful whenever long time-lags need to be captured
-- (information conserved over a prolonged time in the activations).
-- takes 4 input Var objects, returns the required modules and an output Var
-- TODO: add peephole connections
function lstmUnits(size, datain, ingatein, forgetgatein, outgatein)
	-- all the gate inputs get squashed between [0,1]
	ingate = nng.Sigmoid(){ingatein}
	forgetgate = nng.Sigmoid(){forgetgatein}
	outgate = nng.Sigmoid(){outgatein}
	-- data is squashed too, then gated
	newdata = nng.Tanh(){datain}
	statein = nng.CMulTable(){ingate.io.outputs[1], newdata.io.outputs[1]}
	-- the inner "carousel" retains the state information indefinitely
	-- as long as the forgetgate is not used (gated data is added)
	fflop = nng.FlipFlop(size)
	state = nng.CAddTable(){statein.io.outputs[1], fflop.io.outputs[1]}
	nextstate = nng.CMulTable(){forgetgate.io.outputs[1], state.io.outputs[1]}
	fflop:connect(nextstate.io.outputs)
	-- one last squashing, of the output
	preout = nng.CMulTable(){outgate.io.outputs[1], state.io.outputs[1]}
	out = nng.Tanh()(preout.io.outputs)
	
	return {ingate, forgetgate, outgate, newdata, statein, 
			fflop, state, nextstate, preout, out}, out.io.outputs[1]
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


function testLSTM()
	size=2
	datain, ingatein, forgetgatein, outgatein = nng.Var(), nng.Var(), nng.Var(), nng.Var()
	modules, outvar = lstmUnits(size, datain, ingatein, forgetgatein, outgatein)
	g = nng.Graph(unpack(modules))
	print(g, "init")
	
	-- input data: [0.01, 0.1] 
	incs = lab.ones(size)*0.1
	incs[1]= 0.02
	datain:set(incs)
	
	-- gates completely open
	open = lab.ones(size)*1000
	ingatein:set(open) 
	forgetgatein:set(open) 
	outgatein:set(open)
	
	for i=1,10 do
		g:update(outvar)
		print(i, outvar.data[1], outvar.data[2])
		g:tick()
	end	
	print()
	
	-- close input gate on one unit 
	halfopen = lab.ones(size)*1000
	halfopen[1] = -1000
	ingatein:set(halfopen) 
	for i=1,5 do
		g:update(outvar)
		print(i, outvar.data[1], outvar.data[2])
		g:tick()
	end
	print()
	
	-- forget immediately on the other one now 
	closed = lab.ones(size)*-1000
	forgetgatein:set(closed) 
	for i=1,5 do
		g:update(outvar)
		print(i, outvar.data[1], outvar.data[2])
		g:tick()
	end
	
end


--testCounter()
--testFibonacci()
--testElman()
testLSTM()
