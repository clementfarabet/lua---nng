-----------------------------------------------------------------
-- A few basic tests cases (TODO: expand)
-----------------------------------------------------------------

-- require nng lib
require 'nng'

-- Tests to run
tests = {}

-- Check how are asynchronous updates and queries handled.
function tests.validityPropagation()
	local input1 = nng.DataNode()
	local input2 = nng.DataNode()
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
	input1.write(torch.randn(10))
	print(g1(), "valid input1, invalid input2, invalid outputs") --OK
	input2.write(torch.randn(10))
	print(g1(), "valid inputs, invalid outputs") --OK
	output2.read()
	print(g1(), "valid output2, invalid output3") --OK
	output4.read()
	print(g1(), "valid outputs") --OK
	input1.write(torch.randn(10))
	print(g1(), "valid output 2, invalid output3") --OK
	input2.write(torch.randn(10))
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
	local base = nng.DataNode()
	local top = nng.groupNodes({}, base)
	for _=1,depth do
		local affine = nn.Linear(size, size){top.output}
		local squash = nn.Tanh(){affine.output}
		top = nng.groupNodes({top, affine, squash}, squash.output)		
	end
	base.write(torch.ones(size))
	print(top.output.read())
end

function tests.counter()
	local cnode = nng.CounterNode()
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
	local input1 = nng.DataNode()
	local en = nng.ElmanNode({3,9,2}, input1)
	input1.write(torch.zeros(3))
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
	local init = torch.Tensor(2):zero(); init[2] = 1
	local ff = nng.TimeDelayNode(2, init)
	-- the linear transformation does: x,y <- y, x+y 
	local mod = nn.Linear(2,2){ff.output}
	mod.module.weight:fill(1)
	mod.module.weight[1][1]=0 
	mod.module.bias:fill(0)	
	ff.connectInput(mod.output)
	local omod = nn.Sqrt(){mod.output}	
	local fibnode = nng.groupNodes({mod, omod, ff}, omod.output)
	
	-- let's see if the 12th member of the sequence is indeed 12^2...
	for i=1,12 do
		print(i, fibnode.output.read()[1])
		fibnode.tick()
	end
end

-- Illustrate the LSTM gating
function tests.LSTM()
	local size=2
	local datain = nng.DataNode()
	local ingatein = nng.DataNode()
	local forgetgatein = nng.DataNode()
	local outgatein = nng.DataNode()
	local lstmnode = nng.LstmUnit(size, datain, ingatein, forgetgatein, outgatein)
	local outvar = lstmnode.output 
	
	-- input data: [0.01, 0.1] 
	local incs = torch.ones(size)*0.1; incs[1]= 0.02
	datain.write(incs)
	
	-- gates completely open
	local open = torch.ones(size)*1000
	ingatein.write(open) 
	forgetgatein.write(open) 
	outgatein.write(open)
	
	for i=1,10 do
		print(i, outvar.read()[1], outvar.read()[2])
		lstmnode.tick()
	end	
	print()
	
	-- close input gate on one unit 
	local halfopen = torch.ones(size)*1000; halfopen[1] = -1000
	ingatein.write(halfopen) 
	for i=1,5 do
		print(i, outvar.read()[1], outvar.read()[2])
		lstmnode.tick()
	end
	print()
	
	-- forget immediately on the other one now 
	local closed = torch.ones(size)*-1000
	forgetgatein.write(closed) 
	for i=1,5 do
		print(i, outvar.read()[1], outvar.read()[2])
		lstmnode.tick()
	end
end

-- Testing a backward pass
function tests.backward()
	local input = nng.DataNode()
	local sizes = {4, 3, 9, 5, 2}
	local mlp = nng.MultiLayerPerceptron(sizes, input)
	input.write(torch.ones(sizes[1]))
	-- let's do a forward to check the network works
	print("forward", mlp.output.read())
	
	-- now construct the backward layer
	local outerr = nng.DataNode()
	local lasttanh = mlp.nodes[2*(#sizes-1)]
	local firstlinear = mlp.nodes[1]
	local r = nng.groupNodes(nng.backwardTwin(lasttanh, outerr))	
	print(r)
	
	-- see if the backward works
	outerr.write(torch.ones(sizes[#sizes]))
	print("lastback", lasttanh.twin.output.read())
	print("firstback", firstlinear.twin.output.read()[1])
	print("gradients", firstlinear.twin.gradParameters.read()[1])
end

-- Testing a classical ConvNet
function tests.convnet()
	-- define convnet
	local input = nng.DataNode()
	local features = {3, 8, 16, 32}
	local fanins = {1, 4, 16}
	local filters = {7, 7, 7}
	local poolings = {2, 2}
	local convnet = nng.ConvNet(features, fanins, filters, poolings, input)
	
	-- and a linear classifier for a 4-class problem
	local reshaper = nn.Reshape(32){convnet.output}
	local classifier = nn.Linear(32, 4){reshaper.output}
	
	-- loss
	local target = nng.DataNode()
	local logsoftmax = nn.LogSoftMax(){classifier.output}
	local loss = nn.ClassNLLCriterion(){logsoftmax.output, target}
	
	-- random input: a 3-channel 46x46 image
	input.write(torch.randn(3, 46, 46))
	
	-- let's do a forward to check that the network works
	print("forward", logsoftmax.output.read())
	
	-- evaluate the loss
	target.write(3)
	print("target:", target.read())
	print("loss:", loss.output.read())
	
	-- verify the backward construction works too
	nng.backwardTwin(convnet, nng.DataNode())
end

-- Testing criterion
function tests.criterion()
	local input = nng.DataNode()
	local target = nng.DataNode()
	local loss = nn.MSECriterion(){input,target}
	local t = torch.zeros(10); t[4] = 1; -- desired target: 4th class
	input.write(torch.randn(10))
	target.write(t)
	print("output", input.read())
	print("cost", loss.output.read())	
end

-- Test flatten function
function tests.flatten()
	local input = nng.DataNode()
	local sizes = {2, 3, 2}
	local mlp = nng.MultiLayerPerceptron(sizes, input)
	local params = nng.flattenNodes{mlp}
	local nparams = sizes[1]*sizes[2] + sizes[2] + sizes[2]*sizes[3] + sizes[3]
	print('nb of parameters = ' .. nparams)
	mlp.nodes[1].parameters.guts[1]:fill(1)
	mlp.nodes[1].parameters.guts[2]:fill(2)
	mlp.nodes[3].parameters.guts[1]:fill(3)
	mlp.nodes[3].parameters.guts[2]:fill(4)
	print(params)	
end

-- Test weight sharing
function tests.share()
	local input = nng.DataNode()
	local sizes = {2, 3, 2}
	local mlp1 = nng.MultiLayerPerceptron(sizes, input)
	local mlp2 = nng.MultiLayerPerceptron(sizes, input)
	-- share all params btwn two mlps
	nng.shareParameters{mlp1, mlp2}
	-- flatten them
	local flat = nng.flattenNodes{mlp1,mlp2}
	-- set all params of mlp1:
	mlp1.nodes[1].parameters.guts[1]:fill(1)
	mlp1.nodes[1].parameters.guts[2]:fill(2)
	mlp1.nodes[3].parameters.guts[1]:fill(3)
	mlp1.nodes[3].parameters.guts[2]:fill(4)
	-- verify that mlp2's params are good:
	local params = nng.getParameters{mlp2}
	for _,p in ipairs(params) do
		print(p)
	end
	-- print flattened vector
	print(flat)
end

-- Test cloning
function tests.clone()
	local input = nng.DataNode()
	local sizes = {2, 3, 2}
	local mlp1 = nng.MultiLayerPerceptron(sizes, input)
	local mlp2 = nng.cloneNode(mlp1)
	local input2 = mlp2.nodes[1].inputs[1]
	-- let's do a forward with different inputs
	input.write(torch.randn(sizes[1]))
	input2.write(torch.randn(sizes[1]))
	print("forward module1", mlp1.output.read())
	print("forward module2", mlp2.output.read())
	-- and then a forward with the same input
	input2.write(input.read())
	print("forward module1", mlp1.output.read())
	print("forward module2", mlp2.output.read())
		
	-- verify the backward construction is not affected
	nng.backwardTwin(mlp1, nng.DataNode())
	nng.backwardTwin(mlp2, nng.DataNode())
end


-- Test backward with nesting
function tests.backwardNesting()	
	local input = nng.DataNode()
	local mlp1 = nng.MultiLayerPerceptron({3,7,5}, input)
	local mlp2 = nng.MultiLayerPerceptron({5,6,2}, mlp1.output)
	local both = nng.groupNodes({mlp1, mlp2}, mlp2.output)
	print(both)
	input.write(torch.ones(3))	
	print("forward", both.output.read())
	
	local outerr = nng.DataNode()
	nng.backwardTwin(both, outerr)
	outerr.write(torch.ones(2))	
	print("backward", both.twin.output.read())		
end

-- Test non-sequential network structure
function tests.blockConnected()
	local sizes = {{3,4}, {5,7}, {19}, {2,3,2}, {2}}
	local inputs = {}
	for i, s in ipairs(sizes[1]) do
		inputs[i] = nng.DataNode()
	end	
	local net = nng.BlockConnectedPerceptron(sizes, inputs)	
	print(net)	
	for i, input in ipairs(inputs) do
		input.write(torch.ones(input.outputsize))
	end	
	print("forward", net.output.read())	
end

-- Test backward with non-sequential graph
function tests.backwardNonseq()
	local sizes = {{5,3}, {2}, {4,3},{2, 3,2}, {1}}
	local inputs = {}
	for i, s in ipairs(sizes[1]) do
		inputs[i] = nng.DataNode()
	end	
	local net = nng.BlockConnectedPerceptron(sizes, inputs)
	for i, input in ipairs(inputs) do
		input.write(torch.ones(input.outputsize))
	end	
	print("forward", net.output.read())
	
	-- backward
	local outerr = nng.DataNode()
	nng.backwardTwin(net, outerr)	
	outerr.write(torch.ones(net.output.outputsize))		
	print("backward-last", net.nodes[#net.nodes].twin, net.nodes[#net.nodes].twin.output.read())
	print("backward", net.nodes[1].twin, net.nodes[1].twin.output.read()[1])	
end


-- Backward building invoked by adding a criterion
function tests.addCriterion()	
	local input = nng.DataNode()
	local sizes = {2, 3, 2}
	local mlp = nng.MultiLayerPerceptron(sizes, input)
	-- let's say we have an autoencoder, so the output's target equals the input
	local target = input
	local loss, twins = nn.MSECriterion(){mlp.output,target}
	print(twins)
	input.write(torch.randn(2))
	print("output", input.read())
	print("cost", loss.output.read())
	-- as the backward pass was built automatically, we can do this	
	print("inerr1", mlp.nodes[1].twin.output.read())
	print("inerr2", mlp.nodes[2].twin.output.read())
	print("inerr3", mlp.nodes[3].twin.output.read())
end



-- TODO: Test combination of flattening and nesting, and re-flattening, and re-nesting
-- TODO: Test flattening and weight-sharing 
--       what if the shared weights are part of different graphs that are flattened?
-- TODO: Test backward with time-delays
-- TODO: Flattened gradient vector



-- run all the tests
for k,t in pairs(tests) do
	print('==================================================')
	print('testing: ' .. k)
	t()
	print()
end
print('==================================================')
print('All tests done.')
print('==================================================')
	
