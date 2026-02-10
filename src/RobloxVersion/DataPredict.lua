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

local AqwamMachineDeepAndReinforcementLearningLibrary = {}

local Models = script.Models

local Regularizers = script.Regularizers

local Optimizers = script.Optimizers

local ValueSchedulers = script.ValueSchedulers

local GradientClippers = script.GradientClippers

local ExperienceReplays = script.ExperienceReplays

local QuickSetups = script.QuickSetups

local EligibilityTraces = script.EligibilityTraces

local ReinforcementLearningStrategies = script.ReinforcementLearningStrategies

local DistributedTrainingStrategies = script.DistributedTrainingStrategies

local Others = script.Others

AqwamMachineDeepAndReinforcementLearningLibrary.Models = {
	
	-- Regression - 14 Models

	LinearRegression = require(Models.LinearRegression),
	
	QuantileRegression = require(Models.QuantileRegression),
	
	PoissonRegression = require(Models.PoissonRegression),
	
	NegativeBinomialRegression = require(Models.NegativeBinomialRegression),
	
	GammaRegression = require(Models.GammaRegression),
	
	IsotonicRegression = require(Models.IsotonicRegression),
	
	NormalEquationLinearRegression = require(Models.NormalEquationLinearRegression),
	
	BayesianLinearRegression = require(Models.BayesianLinearRegression),
	
	BayesianQuantileLinearRegression = require(Models.BayesianQuantileLinearRegression),
	
	PassiveAggressiveRegressor = require(Models.PassiveAggressiveRegressor),
	
	SupportVectorRegression = require(Models.SupportVectorRegression),
	
	SupportVectorRegressionGradientVariant = require(Models.SupportVectorRegressionGradientVariant),
	
	KNearestNeighboursRegressor = require(Models.KNearestNeighboursRegressor),
	
	RecursiveLeastSquaresRegression = require(Models.RecursiveLeastSquaresRegression),
	
	-- Classification - 13 Models
	
	BinaryRegression = require(Models.BinaryRegression),
	
	PassiveAggressiveClassifier = require(Models.PassiveAggressiveClassifier),
	
	NeuralNetwork = require(Models.NeuralNetwork),
	
	SupportVectorMachine = require(Models.SupportVectorMachine),

	SupportVectorMachineGradientVariant = require(Models.SupportVectorMachineGradientVariant),
	
	NearestCentroid = require(Models.NearestCentroid),
	
	KNearestNeighboursClassifier = require(Models.KNearestNeighboursClassifier),
	
	GaussianNaiveBayes = require(Models.GaussianNaiveBayes),
	
	MultinomialNaiveBayes = require(Models.MultinomialNaiveBayes),
	
	BernoulliNaiveBayes = require(Models.BernoulliNaiveBayes),
	
	ComplementNaiveBayes = require(Models.ComplementNaiveBayes),
	
	CategoricalNaiveBayes = require(Models.CategoricalNaiveBayes),
	
	OrdinalRegression = require(Models.OrdinalRegression),
	
	-- Regression And Classification - 13 Models
	
	IterativeReweightedLeastSquaresRegression = require(Models.IterativeReweightedLeastSquaresRegression),
	
	-- Clustering - 10 Models

	KMeans = require(Models.KMeans),
	
	FuzzyCMeans = require(Models.FuzzyCMeans),
	
	KMedoids = require(Models.KMedoids),
	
	AgglomerativeHierarchical = require(Models.AgglomerativeHierarchical),
	
	ExpectationMaximization = require(Models.ExpectationMaximization),
	
	MeanShift = require(Models.MeanShift),
	
	AffinityPropagation = require(Models.AffinityPropagation),
	
	DensityBasedSpatialClusteringOfApplicationsWithNoise = require(Models.DensityBasedSpatialClusteringOfApplicationsWithNoise),
	
	OrderingPointsToIdentifyClusteringStructure = require(Models.OrderingPointsToIdentifyClusteringStructure),
	
	BisectingCluster = require(Models.BisectingCluster),
	
	-- Deep Reinforcement Learning - 26 Models
	
	DeepQLearning = require(Models.DeepQLearning),
	
	DeepNStepQLearning = require(Models.DeepNStepQLearning),

	DeepDoubleQLearningV1 = require(Models.DeepDoubleQLearningV1),

	DeepDoubleQLearningV2 = require(Models.DeepDoubleQLearningV2),
	
	DeepClippedDoubleQLearning = require(Models.DeepClippedDoubleQLearning),
	
	DeepStateActionRewardStateAction = require(Models.DeepStateActionRewardStateAction),
	
	DeepNStepStateActionRewardStateAction = require(Models.DeepNStepStateActionRewardStateAction),
	
	DeepDoubleStateActionRewardStateActionV1 = require(Models.DeepDoubleStateActionRewardStateActionV1),
	
	DeepDoubleStateActionRewardStateActionV2 = require(Models.DeepDoubleStateActionRewardStateActionV2),
	
	DeepExpectedStateActionRewardStateAction = require(Models.DeepExpectedStateActionRewardStateAction),
	
	DeepNStepExpectedStateActionRewardStateAction = require(Models.DeepNStepExpectedStateActionRewardStateAction),
	
	DeepDoubleExpectedStateActionRewardStateActionV1 = require(Models.DeepDoubleExpectedStateActionRewardStateActionV1),
	
	DeepDoubleExpectedStateActionRewardStateActionV2 = require(Models.DeepDoubleExpectedStateActionRewardStateActionV2),
	
	DeepMonteCarloControl = require(Models.DeepMonteCarloControl),

	DeepOffPolicyMonteCarloControl = require(Models.DeepOffPolicyMonteCarloControl),
	
	DeepTemporalDifference = require(Models.DeepTemporalDifference),
	
	DeepREINFORCE = require(Models.DeepREINFORCE),
	
	VanillaPolicyGradient = require(Models.VanillaPolicyGradient),
	
	ActorCritic = require(Models.ActorCritic),
	
	SoftActorCritic = require(Models.SoftActorCritic),
	
	AdvantageActorCritic = require(Models.AdvantageActorCritic),
	
	TemporalDifferenceActorCritic = require(Models.TemporalDifferenceActorCritic),
	
	ProximalPolicyOptimization = require(Models.ProximalPolicyOptimization),
	
	ProximalPolicyOptimizationClip = require(Models.ProximalPolicyOptimizationClip),
	
	DeepDeterministicPolicyGradient = require(Models.DeepDeterministicPolicyGradient),
	
	TwinDelayedDeepDeterministicPolicyGradient = require(Models.TwinDelayedDeepDeterministicPolicyGradient),
	
	-- Tabular Reinforcement Learning - 17 Models
	
	TabularQLearning = require(Models.TabularQLearning),
	
	TabularNStepQLearning = require(Models.TabularNStepQLearning),
	
	TabularClippedDoubleQLearning = require(Models.TabularClippedDoubleQLearning),
	
	TabularDoubleQLearningV1 = require(Models.TabularDoubleQLearningV1),
	
	TabularDoubleQLearningV2 = require(Models.TabularDoubleQLearningV2),

	TabularStateActionRewardStateAction = require(Models.TabularStateActionRewardStateAction),
	
	TabularNStepStateActionRewardStateAction = require(Models.TabularNStepStateActionRewardStateAction),
	
	TabularDoubleStateActionRewardStateActionV1 = require(Models.TabularDoubleStateActionRewardStateActionV1),

	TabularDoubleStateActionRewardStateActionV2 = require(Models.TabularDoubleStateActionRewardStateActionV2),
	
	TabularExpectedStateActionRewardStateAction = require(Models.TabularExpectedStateActionRewardStateAction),
	
	TabularNStepExpectedStateActionRewardStateAction = require(Models.TabularNStepExpectedStateActionRewardStateAction),
	
	TabularDoubleExpectedStateActionRewardStateActionV1 = require(Models.TabularDoubleExpectedStateActionRewardStateActionV1),

	TabularDoubleExpectedStateActionRewardStateActionV2 = require(Models.TabularDoubleExpectedStateActionRewardStateActionV2),
	
	TabularMonteCarloControl = require(Models.TabularMonteCarloControl),
	
	TabularOffPolicyMonteCarloControl = require(Models.TabularOffPolicyMonteCarloControl),
	
	TabularTemporalDifference = require(Models.TabularTemporalDifference),
	
	TabularREINFORCE = require(Models.TabularREINFORCE),
	
	-- Sequence Modelling - 3 Models
	
	Markov = require(Models.Markov),
	
	DynamicBayesianNetwork = require(Models.DynamicBayesianNetwork),
	
	ConditionalRandomField = require(Models.ConditionalRandomField),

	-- Filtering - 4 Models
	
	KalmanFilter = require(Models.KalmanFilter),
	
	ExtendedKalmanFilter = require(Models.ExtendedKalmanFilter),
	
	UnscentedKalmanFilter = require(Models.UnscentedKalmanFilter),
	
	UnscentedKalmanFilterDataPredictVariant = require(Models.UnscentedKalmanFilterDataPredictVariant),

	-- Outlier Detection - 4 Models
	
	OneClassPassiveAggressiveClassifier = require(Models.OneClassPassiveAggressiveClassifier),
	
	OneClassSupportVectorMachine = require(Models.OneClassSupportVectorMachine),
	
	LocalOutlierFactor = require(Models.LocalOutlierFactor),

	LocalOutlierProbability = require(Models.LocalOutlierProbability),
	
	-- Recommendation - 5 Models

	FactorizationMachine = require(Models.FactorizationMachine),
	
	FactorizedPairwiseInteraction = require(Models.FactorizedPairwiseInteraction),
	
	SimonFunkMatrixFactorization = require(Models.SimonFunkMatrixFactorization),
	
	SimonFunkMatrixFactorizationWithBiases = require(Models.SimonFunkMatrixFactorizationWithBiases),
	
	TwoTower = require(Models.TwoTower),
	
	-- Generative - 4 Models

	GenerativeAdversarialNetwork = require(Models.GenerativeAdversarialNetwork),
	
	ConditionalGenerativeAdversarialNetwork = require(Models.ConditionalGenerativeAdversarialNetwork),

	WassersteinGenerativeAdversarialNetwork = require(Models.WassersteinGenerativeAdversarialNetwork),

	ConditionalWassersteinGenerativeAdversarialNetwork = require(Models.ConditionalWassersteinGenerativeAdversarialNetwork),
	
	-- Feature-Class Containers - 1 Model
	
	Table = require(Models.Table),

}

AqwamMachineDeepAndReinforcementLearningLibrary.Regularizers = {
	
	ElasticNet = require(Regularizers.ElasticNet),
	
	Lasso = require(Regularizers.Lasso),
	
	Ridge = require(Regularizers.Ridge),
	
}

AqwamMachineDeepAndReinforcementLearningLibrary.Optimizers = {

	AdaptiveDelta = require(Optimizers.AdaptiveDelta),
	
	AdaptiveFactor = require(Optimizers.AdaptiveFactor),
	
	AdaptiveGradient = require(Optimizers.AdaptiveGradient),

	AdaptiveMomentEstimation = require(Optimizers.AdaptiveMomentEstimation),

	AdaptiveMomentEstimationMaximum = require(Optimizers.AdaptiveMomentEstimationMaximum),
	
	AdaptiveMomentEstimationWeightDecay = require(Optimizers.AdaptiveMomentEstimationWeightDecay),
	
	Gravity = require(Optimizers.Gravity),
	
	Momentum = require(Optimizers.Momentum),

	NesterovAcceleratedAdaptiveMomentEstimation = require(Optimizers.NesterovAcceleratedAdaptiveMomentEstimation),
	
	RectifiedAdaptiveMomentEstimation = require(Optimizers.RectifiedAdaptiveMomentEstimation),
	
	ResilientBackwardPropagation = require(Optimizers.ResilientBackwardPropagation),
	
	RootMeanSquarePropagation = require(Optimizers.RootMeanSquarePropagation),

}

AqwamMachineDeepAndReinforcementLearningLibrary.ValueSchedulers = {
	
	Chained = require(ValueSchedulers.Chained),
	
	Constant = require(ValueSchedulers.Constant),
	
	CosineAnnealing = require(ValueSchedulers.CosineAnnealing),
	
	Exponential = require(ValueSchedulers.Exponential),
	
	InverseSquareRoot = require(ValueSchedulers.InverseSquareRoot),
	
	InverseTime = require(ValueSchedulers.InverseTime),
	
	Linear = require(ValueSchedulers.Linear),
	
	MultipleStep = require(ValueSchedulers.MultipleStep),
	
	Multiplicative = require(ValueSchedulers.Multiplicative),
	
	Polynomial = require(ValueSchedulers.Polynomial),
	
	Sequential = require(ValueSchedulers.Sequential),

	Step = require(ValueSchedulers.Step),

}

AqwamMachineDeepAndReinforcementLearningLibrary.GradientClippers = {

	ClipValue = require(GradientClippers.ClipValue),

	ClipNormalization = require(GradientClippers.ClipNormalization),
	
}

AqwamMachineDeepAndReinforcementLearningLibrary.ExperienceReplays = {

	UniformExperienceReplay = require(ExperienceReplays.UniformExperienceReplay),

	PrioritizedExperienceReplay = require(ExperienceReplays.PrioritizedExperienceReplay),

	NStepExperienceReplay = require(ExperienceReplays.NStepExperienceReplay),

}

AqwamMachineDeepAndReinforcementLearningLibrary.QuickSetups = {

	SingleCategoricalPolicy = require(QuickSetups.SingleCategoricalPolicy),

	SingleDiagonalGaussianPolicy = require(QuickSetups.SingleDiagonalGaussianPolicy),
	
	QueuedCategoricalPolicy = require(QuickSetups.QueuedCategoricalPolicy),

	QueuedDiagonalGaussianPolicy = require(QuickSetups.QueuedDiagonalGaussianPolicy),
	
	ParallelCategoricalPolicy = require(QuickSetups.ParallelCategoricalPolicy),

	ParallelDiagonalGaussianPolicy = require(QuickSetups.ParallelDiagonalGaussianPolicy),

}

AqwamMachineDeepAndReinforcementLearningLibrary.EligibilityTraces = {
	
	AccumulatingTrace = require(EligibilityTraces.AccumulatingTrace),
	
	ReplacingTrace = require(EligibilityTraces.ReplacingTrace),
	
	DutchTrace = require(EligibilityTraces.DutchTrace),
	
}

AqwamMachineDeepAndReinforcementLearningLibrary.ReinforcementLearningStrategies = {
	
	RandomNetworkDistillation = require(ReinforcementLearningStrategies.RandomNetworkDistillation),
	
	GenerativeAdversarialImitationLearning = require(ReinforcementLearningStrategies.GenerativeAdversarialImitationLearning),

	WassersteinGenerativeAdversarialImitationLearning = require(ReinforcementLearningStrategies.WassersteinGenerativeAdversarialImitationLearning),
	
}

AqwamMachineDeepAndReinforcementLearningLibrary.DistributedTrainingStrategies = {
	
	DistributedGradientsCoordinator = require(DistributedTrainingStrategies.DistributedGradientsCoordinator),

	DistributedModelParametersCoordinator = require(DistributedTrainingStrategies.DistributedModelParametersCoordinator),
	
}

AqwamMachineDeepAndReinforcementLearningLibrary.Others = {
	
	ModelTrainingModifier = require(Others.ModelTrainingModifier),
	
	ModelSafeguardWrapper = require(Others.ModelSafeguardWrapper),
	
	ModelParametersMerger = require(Others.ModelParametersMerger),

	ModelChecker = require(Others.ModelChecker),
	
	OneVsAll = require(Others.OneVsAll),
	
	OneVsOne = require(Others.OneVsOne),

	OnlineLearning = require(Others.OnlineLearning),
	
	DatasetCreator = require(Others.DatasetCreator),
	
	ConfusionMatrixCreator = require(Others.ConfusionMatrixCreator),

}

return AqwamMachineDeepAndReinforcementLearningLibrary	parameterDictionary = parameterDictionary or {}

	local NewRecursiveLeastSquaresRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewRecursiveLeastSquaresRegressionModel, RecursiveLeastSquaresRegressionModel)

	NewRecursiveLeastSquaresRegressionModel:setName("RecursiveLeastSquaresRegression")
	
	NewRecursiveLeastSquaresRegressionModel.lossFunction = parameterDictionary.lossFunction or defaultLossFunction
	
	NewRecursiveLeastSquaresRegressionModel.forgetFactor = parameterDictionary.forgetFactor or defaultForgetFactor
	
	NewRecursiveLeastSquaresRegressionModel.useLogProbabilities = NewRecursiveLeastSquaresRegressionModel:getValueOrDefaultValue(parameterDictionary.useLogProbabilities, defaultUseLogProbabilities)

	return NewRecursiveLeastSquaresRegressionModel
	
end

function RecursiveLeastSquaresRegressionModel:train(featureMatrix, labelVector)

	local numberOfData = #featureMatrix

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local numberOfFeatures = #featureMatrix[1]
	
	local lossFunction = self.lossFunction
	
	local lossFunctionToApply = lossFunctionList[lossFunction]

	if (not lossFunctionToApply) then error("Invalid loss function.") end
	
	local forgetFactor = self.forgetFactor
	
	local ModelParameters = self.ModelParameters or {}
	
	local weightVector = ModelParameters[1] or self:initializeMatrixBasedOnMode({numberOfFeatures, 1})
	
	if (numberOfFeatures ~= #weightVector) then error("The number of features are not the same as the model parameters.") end
	
	local errorCovarianceMatrix = ModelParameters[2] or AqwamTensorLibrary:createIdentityTensor({numberOfFeatures, numberOfFeatures})
	
	local featureVector
	
	local predictedValue
	
	local lossValue
	
	local kalmanGainVectorNumerator
	
	local transposedFeatureVector
	
	local kalmanGainVectorDenominator
	
	local kalmanGainVector
	
	local transposedKalmanGainVector
	
	local weightChangeVector
	
	local cost = 0
	
	for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		featureVector = {unwrappedFeatureVector}
		
		predictedValue = AqwamTensorLibrary:dotProduct(featureVector, weightVector)[1][1]
		
		lossValue = predictedValue - labelVector[dataIndex][1]
		
		kalmanGainVectorNumerator = AqwamTensorLibrary:dotProduct(featureVector, errorCovarianceMatrix) -- 1 x n
		
		transposedFeatureVector = AqwamTensorLibrary:transpose(featureVector) -- n x 1
		
		kalmanGainVectorDenominator = AqwamTensorLibrary:dotProduct(featureVector, errorCovarianceMatrix, transposedFeatureVector) -- 1 x 1
		
		kalmanGainVectorDenominator = AqwamTensorLibrary:add(forgetFactor, kalmanGainVectorDenominator)
		
		kalmanGainVector = AqwamTensorLibrary:divide(kalmanGainVectorNumerator, kalmanGainVectorDenominator) -- 1 x n
		
		transposedKalmanGainVector = AqwamTensorLibrary:transpose(kalmanGainVector)
		
		weightChangeVector = AqwamTensorLibrary:multiply(kalmanGainVector, lossValue) -- 1 x n
		
		weightChangeVector = AqwamTensorLibrary:transpose(weightChangeVector)

		weightVector = AqwamTensorLibrary:add(weightVector, weightChangeVector)
		
		errorCovarianceMatrix = AqwamTensorLibrary:subtract(errorCovarianceMatrix, AqwamTensorLibrary:dotProduct(transposedKalmanGainVector, featureVector, errorCovarianceMatrix))

		if (forgetFactor ~= 1) then errorCovarianceMatrix = AqwamTensorLibrary:divide(errorCovarianceMatrix, forgetFactor) end
		
		cost = cost + lossFunctionToApply(lossValue)
		
	end
	
	self.ModelParameters = {weightVector, errorCovarianceMatrix}
	
	cost = cost / numberOfData

	return {cost}

end

function RecursiveLeastSquaresRegressionModel:predict(featureMatrix, thresholdMatrix)

	if (thresholdMatrix) then

		if (#featureMatrix ~= #thresholdMatrix) then error("The feature matrix and the threshold matrix does not contain the same number of rows.") end

	end

	local ModelParameters = self.ModelParameters

	local weightVector

	local covarianceMatrix

	if (not ModelParameters) then

		local numberOfFeatures = #featureMatrix[1]

		weightVector = self:initializeMatrixBasedOnMode({numberOfFeatures, 1})

		covarianceMatrix = AqwamTensorLibrary:createIdentityTensor({numberOfFeatures, numberOfFeatures})

		self.ModelParameters = {weightVector, covarianceMatrix}

	else

		weightVector = ModelParameters[1]

		covarianceMatrix = ModelParameters[2]

	end

	local predictedVector = AqwamTensorLibrary:dotProduct(featureMatrix, weightVector)

	if (not thresholdMatrix) then return predictedVector end

	local transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)

	local predictedVarianceVectorPart1 = AqwamTensorLibrary:dotProduct(featureMatrix, covarianceMatrix)

	local predictedVarianceVectorPart2 = AqwamTensorLibrary:dotProduct(predictedVarianceVectorPart1, transposedFeatureMatrix)

	local predictedVarianceVector = {}

	for i, predictedVarianceTable in ipairs(predictedVarianceVectorPart2) do

		predictedVarianceVector[i] = {predictedVarianceTable[i]}

	end

	local predictedStandardDeviationVector = AqwamTensorLibrary:applyFunction(math.sqrt, predictedVarianceVector)

	local probabilityMatrix = calculateGaussianProbability(self.useLogProbabilities, thresholdMatrix, predictedVector, predictedStandardDeviationVector)

	return predictedVector, probabilityMatrix 

end

return RecursiveLeastSquaresRegressionModel
