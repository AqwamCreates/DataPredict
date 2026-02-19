--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseSolver = require(script.Parent.BaseSolver)

local GaussSeidelSolver = {}

GaussSeidelSolver.__index = GaussSeidelSolver

setmetatable(GaussSeidelSolver, BaseSolver)

local function rearrangeMatrixToDominantDiagonalMatrix(matrix)

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(matrix)

	local numberOfRows = dimensionSizeArray[1]

	local numberOfColumns = dimensionSizeArray[2]

	local dominantDiagonalMatrix = {}

	local diagonaMatrixIndexArray = {}

	local duplicateValueBlacklistArray = {}

	local maximumAbsoluteValue

	local columnIndexWithMaximumValue

	local absoluteValue

	local sumAbsoluteValue

	for rowIndex, unwrappedVector in ipairs(matrix) do

		maximumAbsoluteValue = 0

		columnIndexWithMaximumValue = nil

		for primaryColumnIndex, value in ipairs(unwrappedVector) do

			absoluteValue = math.abs(value)

			sumAbsoluteValue = 0

			for secondaryColumnIndex, value in ipairs(unwrappedVector) do

				if (secondaryColumnIndex ~= primaryColumnIndex) then sumAbsoluteValue = sumAbsoluteValue + math.abs(value) end

			end

			if (absoluteValue > sumAbsoluteValue) then

				if (absoluteValue > maximumAbsoluteValue) then

					maximumAbsoluteValue = absoluteValue

					columnIndexWithMaximumValue = primaryColumnIndex

				end

			end

		end

		diagonaMatrixIndexArray[rowIndex] = columnIndexWithMaximumValue

	end

	for i, value in ipairs(diagonaMatrixIndexArray) do

		if (table.find(duplicateValueBlacklistArray, value)) then return matrix end

		table.insert(duplicateValueBlacklistArray, value)

	end

	for rowIndex, unwrappedVector in ipairs(matrix) do

		columnIndexWithMaximumValue = diagonaMatrixIndexArray[rowIndex]

		dominantDiagonalMatrix[columnIndexWithMaximumValue] = unwrappedVector

	end

	return dominantDiagonalMatrix, diagonaMatrixIndexArray

end

function GaussSeidelSolver.new(parameterDictionary)
	
	local NewGaussSeidelSolver = BaseSolver.new(parameterDictionary)
	
	setmetatable(NewGaussSeidelSolver, GaussSeidelSolver)
	
	NewGaussSeidelSolver:setName("GaussSeidel")
	
	NewGaussSeidelSolver:setCalculateFunction(function(weightMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		-- Can only cache from linear models since the derivative is a feature matrix. Hence, these values are constant.
		
		local isNonLinearInput = NewGaussSeidelSolver.isNonLinearInput
		
		local isLinearInput = (not isNonLinearInput)
		
		local cache = NewGaussSeidelSolver.cache or {}
		
		local transposedFirstDerivativeMatrix = (isLinearInput and cache[1]) or AqwamTensorLibrary:transpose(firstDerivativeMatrix)
		
		local aMatrix = (isLinearInput and cache[2])
		
		local inverseLMatrix = cache[3]
		
		local uMatrix = cache[4]
		
		local diagonaMatrixIndexArray = cache[5]
		
		local weightMatrixDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightMatrix)

		local numberOfFeatures = weightMatrixDimensionSizeArray[1]

		local numberOfOutputs = weightMatrixDimensionSizeArray[2]
		
		local bMatrix = AqwamTensorLibrary:dotProduct(transposedFirstDerivativeMatrix, firstDerivativeLossMatrix)
		
		local lMatrix

		if (not aMatrix) then

			local transposedFirstDerivativeMatrix = AqwamTensorLibrary:transpose(firstDerivativeMatrix)

			aMatrix = AqwamTensorLibrary:dotProduct(transposedFirstDerivativeMatrix, firstDerivativeMatrix)
			
			aMatrix, diagonaMatrixIndexArray = rearrangeMatrixToDominantDiagonalMatrix(aMatrix)
			
			if (isLinearInput) then 
				
				cache[2] = aMatrix 
				
				cache[5] = diagonaMatrixIndexArray
				
			end

		end
		
		if (inverseLMatrix) then
			
			if (not isLinearInput) then 
				
				lMatrix = AqwamTensorLibrary:multiply(lMatrix, aMatrix)

				inverseLMatrix = AqwamTensorLibrary:inverse(lMatrix)
				
			end
		
		else
			
			lMatrix = AqwamTensorLibrary:createLowerTriangularTensor(weightMatrixDimensionSizeArray, 1)
			
			if (isLinearInput) then

				lMatrix = AqwamTensorLibrary:multiply(lMatrix, aMatrix)
				
				inverseLMatrix = AqwamTensorLibrary:inverse(lMatrix)

				cache[3] = lMatrix

			else

				cache[3] = lMatrix

				lMatrix = AqwamTensorLibrary:multiply(lMatrix, aMatrix)
				
				inverseLMatrix = AqwamTensorLibrary:inverse(lMatrix)

			end
			
		end
		
		if (uMatrix) then
			
			if (not isLinearInput) then
				
				uMatrix = AqwamTensorLibrary:multiply(uMatrix, aMatrix)
				
				uMatrix = AqwamTensorLibrary:dotProduct(uMatrix, weightMatrix)
				
			end
			
		else
			
			uMatrix = AqwamTensorLibrary:createUpperTriangularTensor(weightMatrixDimensionSizeArray, 0, 1)
			
			if (isLinearInput) then
				
				uMatrix = AqwamTensorLibrary:multiply(uMatrix, aMatrix)
				
				uMatrix = AqwamTensorLibrary:dotProduct(uMatrix, weightMatrix)
				
				cache[4] = uMatrix
				
			else
				
				cache[4] = uMatrix
				
				uMatrix = AqwamTensorLibrary:multiply(uMatrix, aMatrix)
				
				uMatrix = AqwamTensorLibrary:dotProduct(uMatrix, weightMatrix)
				
			end
			
		end
		
		local newWeightMatrix = AqwamTensorLibrary:subtract(bMatrix, uMatrix)

		newWeightMatrix = AqwamTensorLibrary:dotProduct(inverseLMatrix, newWeightMatrix)
		
		if (diagonaMatrixIndexArray) then

			newWeightMatrix = AqwamTensorLibrary:permute(newWeightMatrix, diagonaMatrixIndexArray)

		end
		
		NewGaussSeidelSolver.cache = cache
		
		return AqwamTensorLibrary:subtract(newWeightMatrix, weightMatrix)
		
	end)
	
	return NewGaussSeidelSolver
	
end

return GaussSeidelSolver
