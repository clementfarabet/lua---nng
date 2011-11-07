
require 'nng'

function testValidityPropagation()
	-- Check how are asynchronous updates and queries handled.
	input1 = nng.Var()
	input2 = nng.Var()
	mod1 = nng.Linear(10,10){input1}
	mod2 = nng.Linear(10,10){input2}
	mod3 = nng.Tanh()(mod1.io.outputs, mod2.io.outputs)
	output2 = mod2.io.outputs[1]
	output3 = mod3.io.outputs[1]
	g1 = nng.Graph(mod1,mod2,mod3)
	
	print(g1, "invalid inputs, invalid outputs") --OK
	input1:set(lab.randn(10))
	print(g1, "valid input1, invalid input2, invalid outputs") --OK
	input2:set(lab.randn(10))
	print(g1, "valid inputs, invalid outputs") --OK
	g1:update(output2)
	print(g1, "valid output2, invalid output3") --OK
	g1:update(output3)
	print(g1, "valid outputs") --OK
	input1:set(lab.randn(10))
	print(g1, "valid output 2, invalid output3") --OK
	input2:set(lab.randn(10))
	print(g1, "invalid outputs") --OK
	g1:update()
	print(g1, "valid outputs") --OK
end



testValidityPropagation()
