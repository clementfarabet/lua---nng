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
	local input = g.DataNode()
	local sizes = {4, 3, 9, 5, 2}
	local mlp = g.MultiLayerPerceptron(sizes, input)
	input.write(lab.ones(sizes[1]))
	-- let's do a forward to check the network works
	print("forward", mlp.output.read())
	
	-- now construct the backward layer
	local outerr = g.DataNode()
	local lasttanh = mlp.nodes[2*(#sizes-1)]
	local firstlinear = mlp.nodes[1]
	local r = g.groupNodes(g.backwardTwin(lasttanh, outerr))	
	print(r)
	
	-- see if the backward works
	outerr.write(lab.ones(sizes[#sizes]))
	print("lastback", lasttanh.twin.output.read())
	print("firstback", firstlinear.twin.output.read()[1])
	print("gradients", firstlinear.twin.gradParameters.read()[1])
end

-- Testing a classical ConvNet
function tests.convnet()
	-- define convnet
	input = g.DataNode()
	features = {3, 8, 16, 32}
	fanins = {1, 4, 16}
	filters = {7, 7, 7}
	poolings = {2, 2}
	convnet = g.ConvNet(features, fanins, filters, poolings, input)
	
	-- and a linear classifier for a 4-class problem
	reshaper = nn.Reshape(32){convnet.output}
	classifier = nn.Linear(32, 4){reshaper.output}
	
	-- loss
	target = g.DataNode()
	logsoftmax = nn.LogSoftMax(){classifier.output}
	loss = nn.ClassNLLCriterion(){logsoftmax.output, target}
	
	-- random input: a 3-channel 46x46 image
	input.write(lab.randn(3, 46, 46))
	
	-- let's do a forward to check that the network works
	print("forward", logsoftmax.output.read())
	
	-- evaluate the loss
	target.write(3)
	print("target:", target.read())
	print("loss:", loss.output.read())
end

-- Testing criterion
function tests.criterion()
	input = g.DataNode()
	target = g.DataNode()
	mlp = g.MultiLayerPerceptron({10,2}, input)
	loss = nn.MSECriterion(){input,target}
	t = lab.zeros(10); t[4] = 1; -- desired target: 4th class
	input.write(lab.randn(10))
	target.write(t)
	print("output", mlp.output.read())
	print("cost", loss.output.read())
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

-- Test cloning
function tests.clone()
	input = g.DataNode()
	sizes = {2, 3, 2}
	mlp1 = g.MultiLayerPerceptron(sizes, input)
	mlp2 = g.cloneNode(mlp1)
	input2 = mlp2.nodes[1].inputs[1]
	-- let's do a forward with different inputs
	input.write(lab.randn(sizes[1]))
	input2.write(lab.randn(sizes[1]))
	print("forward module1", mlp1.output.read())
	print("forward module2", mlp2.output.read())
	-- and then a forward with the same input
	input2.write(input.read())
	print("forward module1", mlp1.output.read())
	print("forward module2", mlp2.output.read())
end


-- Test backward with nesting
function tests.backwardNesting()	
	local input = g.DataNode()
	local mlp1 = g.MultiLayerPerceptron({3,7,5}, input)
	local mlp2 = g.MultiLayerPerceptron({5,6,2}, mlp1.output)
	local both = g.groupNodes({mlp1, mlp2}, mlp2.output)
	print(both)
	input.write(lab.ones(3))	
	print("forward", both.output.read())
	
	local outerr = g.DataNode()
	g.backwardTwin(both, outerr)
	outerr.write(lab.ones(2))	
	print("backward", both.twin.output.read())		
end

-- TODO: test backward with non-sequential graph
function tests.backwardNonseq()
	local sizes = {4, 5, 2}
	
	-- we can start on the output
	local outerr = g.DataNode()
	local input = g.DataNode()
	local mlp = g.MultiLayerPerceptron(sizes, input)
	local x = g.backwardTwin(mlp.output, outerr)
	input.write(lab.ones(sizes[1]))	
	print("forward", mlp.output.read()[1])
	outerr.write(lab.ones(sizes[#sizes]))		
	print("backward-last", mlp.nodes[#mlp.nodes].twin.output.read()[1])
	print("backward", mlp.nodes[1].twin.output.read()[1])
	
	-- or call it on the grouping node
	local outerr = g.DataNode()
	local input = g.DataNode()
	local mlp = g.MultiLayerPerceptron(sizes, input)
	local x = g.backwardTwin(mlp, outerr)
	input.write(lab.ones(sizes[1]))
	print("forward", mlp.output.read()[1])
	outerr.write(lab.ones(sizes[#sizes]))
	print("backward-last", mlp.nodes[#mlp.nodes].twin.output.read()[1])
	print("backward", mlp.nodes[1].twin.output.read()[1])		
end

-- TODO: Test combination of flattening and nesting, and re-flattening, and re-nesting
-- TODO: Test flattening and weight-sharing 
--       what if the shared weights are part of different graphs that are flattened?
-- TODO: Test backward with time-delays
-- TODO: Flattened gradient vector
-- TODO: Backward building 	invoked by adding a criterion


-- run all the tests
for k,t in pairs(tests) do
	print('==================================================')
	print('testing: ' .. k)
	t()
	print()
end
