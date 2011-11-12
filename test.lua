-----------------------------------------------------------------
-- A few basic tests cases (TODO: expand)
-----------------------------------------------------------------

-- dependencies
require('g')

-- Tests to run
tests = {}

-- Check how are asynchronous updates and queries handled.
function tests.validityPropagation()
	local input1 = g.DataNode()
	local input2 = g.DataNode()
	local mod1 = nn.Linear(10,10){input1}
	local mod2 = nn.Linear(10,10){input2}
	local mod3 = nn.JoinTable(1){mod1.output, mod2.output}		
	local mod4 = nn.Tanh(){mod3.output}
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
function tests.nesting()
	local depth = 100
	local size = 5
	local base = g.DataNode()
	local top = g.groupNodes({}, base)
	for _=1,depth do
		local affine = nn.Linear(size, size){top.output}
		local squash = nn.Tanh(){affine.output}
		top = g.groupNodes({top, affine, squash}, squash.output)		
	end
	base.write(lab.ones(size))
	print(top.output.read())
end

function tests.counter()
	local cnode = g.CounterNode()
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
function tests.elman()
	local input1 = g.DataNode()
	local en = g.ElmanNode({3,9,2}, input1)
	input1.write(lab.zeros(3))
	print(en.output.read()[1], "first pass")
	print(en.output.read()[1], "first pass (still)")
	en.tick()
	print(en.output.read()[1], "second pass")
	en.tick()
	print(en.output.read()[1], "third pass")		
end

-- A simple neural network that produces the square roots of the Fibonacci numbers	
function tests.fibonacci()
	-- the flipflop is initialized with (0,1)
	local init = torch.Tensor(2):zero()
	init[2] = 1
	local ff = g.TimeDelayNode(2, init)
	-- the linear transformation does: x,y <- y, x+y 
	local mod = nn.Linear(2,2){ff.output}
	mod.module.weight:fill(1)
	mod.module.weight[1][1]=0 
	mod.module.bias:fill(0)	
	ff.connectInput(mod.output)
	local omod = nn.Sqrt(){mod.output}	
	local fibnode = g.groupNodes({mod, omod, ff}, omod.output)
	
	-- let's see if the 12th member of the sequence is indeed 12^2...
	for i=1,12 do
		print(i, fibnode.output.read()[1])
		fibnode.tick()
	end
end

-- Illustrate the LSTM gating
function tests.LSTM()
	local size=2
	local datain = g.DataNode()
	local ingatein = g.DataNode()
	local forgetgatein = g.DataNode()
	local outgatein = g.DataNode()
	local lstmnode = g.LstmUnit(size, datain, ingatein, forgetgatein, outgatein)
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

-- Testing a backward pass
function tests.backward()
	input = g.DataNode()
	sizes = {4, 3, 9, 5, 2}
	mlp = g.MultiLayerPerceptron(sizes, input)
	input.write(lab.ones(sizes[1]))
	-- let's do a forward to check the network works
	print("forward", mlp.output.read())
	
	-- now construct the backward layer
	outerr = g.DataNode()
	lasttanh = mlp.nodes[2*(#sizes-1)]
	firstlinear = mlp.nodes[1]
	r = g.groupNodes(g.backwardTwin(lasttanh, outerr))	
	print(r)
	
	-- see if the backward works
	outerr.write(lab.ones(sizes[#sizes]))
	print("lastback", lasttanh.twin.output.read())
	print("firstback", firstlinear.output.read())
	print("gradients", firstlinear.twin.gradParameters.read()[1])
end

-- Test flatten function
function tests.flatten()
	input = g.DataNode()
	sizes = {2, 3, 2}
	mlp = g.MultiLayerPerceptron(sizes, input)
	params = g.flattenNodes{mlp}
	nparams = sizes[1]*sizes[2] + sizes[2] + sizes[2]*sizes[3] + sizes[3]
	print('nb of parameters = ' .. nparams)
	mlp.nodes[1].parameters.guts[1]:fill(1)
	mlp.nodes[1].parameters.guts[2]:fill(2)
	mlp.nodes[3].parameters.guts[1]:fill(3)
	mlp.nodes[3].parameters.guts[2]:fill(4)
	print(params)
end

-- Test weight sharing
function tests.share()
	input = g.DataNode()
	sizes = {2, 3, 2}
	mlp1 = g.MultiLayerPerceptron(sizes, input)
	mlp2 = g.MultiLayerPerceptron(sizes, input)
	-- share all params btwn two mlps
	g.shareParameters{mlp1, mlp2}
	-- flatten them
	flat = g.flattenNodes{mlp1,mlp2}
	-- set all params of mlp1:
	mlp1.nodes[1].parameters.guts[1]:fill(1)
	mlp1.nodes[1].parameters.guts[2]:fill(2)
	mlp1.nodes[3].parameters.guts[1]:fill(3)
	mlp1.nodes[3].parameters.guts[2]:fill(4)
	-- verify that mlp2's params are good:
	params = g.getParameters{mlp2}
	for _,p in ipairs(params) do
		print(p)
	end
	-- print flattened vector
	print(flat)
end

-- run all the tests
for k,t in pairs(tests) do
	print('==================================================')
	print('testing: ' .. k)
	t()
	print()
end
