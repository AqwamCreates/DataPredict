function LogisticRegressionOneVsAllModel:predict(featureMatrix)
	
	local highestClass
	
	local softmax

	local highestSoftmax = -math.huge
	
	local zVector = AqwamMatrixLibrary:dotProduct(featureMatrix, self.ModelParameters)
	
	local softmaxVector = AqwamMatrixLibrary:applyFunction(math.exp, zVector)
	
	local softmaxSumVector = AqwamMatrixLibrary:sum(softmaxVector)
	
	for column = 1, #softmaxVector[1], 1 do
		
		softmax = softmaxVector[1][column]
		
		if (softmax > highestSoftmax) then
			
			highestClass = self.ClassesList[column]
			
			highestSoftmax = softmax
			
		end
		
	end
	
	if (softmaxSumVector ~= math.huge) then
		
		highestSoftmax = highestSoftmax / softmaxSumVector
		
	else
		
		highestSoftmax = 1.0
		
	end
	
	return highestClass, highestSoftmax
	
end


