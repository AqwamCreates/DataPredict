local AqwamMachineAndDeepLearningLibrary = {}

local Models = script.Models

local Optimizers = script.Optimizers

local ValueSchedulers = script.ValueSchedulers

local ExperienceReplays = script.ExperienceReplays

local QuickSetups = script.QuickSetups

local ReinforcementLearningStrategies = script.ReinforcementLearningStrategies

local DistributedTrainingStrategies = script.DistributedTrainingStrategies

local Others = script.Others

AqwamMachineAndDeepLearningLibrary.Models = {

	LinearRegression = require(Models.LinearRegression),
	
	NormalLinearRegression = require(Models.NormalLinearRegression),
	
	LogisticRegression = require(Models.LogisticRegression),
	
	NeuralNetwork = require(Models.NeuralNetwork),
	
	SupportVectorMachine = require(Models.SupportVectorMachine),
	
	KNearestNeighbours = require(Models.KNearestNeighbours),
	
	GaussianNaiveBayes = require(Models.GaussianNaiveBayes),
	
	MultinomialNaiveBayes = require(Models.MultinomialNaiveBayes),
	
	BernoulliNaiveBayes = require(Models.BernoulliNaiveBayes),
	
	KMeans = require(Models.KMeans),
	
	ExpectationMaximization = require(Models.ExpectationMaximization),
	
	AgglomerativeHierarchical = require(Models.AgglomerativeHierarchical),
	
	MeanShift = require(Models.MeanShift),
	
	DensityBasedSpatialClusteringOfApplicationsWithNoise = require(Models.DensityBasedSpatialClusteringOfApplicationsWithNoise),
	
	KMedoids = require(Models.KMedoids),
	
	AffinityPropagation = require(Models.AffinityPropagation),
	
	DeepQLearning = require(Models.DeepQLearning),

	DeepDoubleQLearningV1 = require(Models.DeepDoubleQLearningV1),

	DeepDoubleQLearningV2 = require(Models.DeepDoubleQLearningV2),
	
	DeepClippedDoubleQLearning = require(Models.DeepClippedDoubleQLearning),
	
	DeepStateActionRewardStateAction = require(Models.DeepStateActionRewardStateAction),
	
	DeepDoubleStateActionRewardStateActionV1 = require(Models.DeepDoubleStateActionRewardStateActionV1),
	
	DeepDoubleStateActionRewardStateActionV2 = require(Models.DeepDoubleStateActionRewardStateActionV2),
	
	DeepExpectedStateActionRewardStateAction = require(Models.DeepExpectedStateActionRewardStateAction),
	
	DeepDoubleExpectedStateActionRewardStateActionV1 = require(Models.DeepDoubleExpectedStateActionRewardStateActionV1),
	
	DeepDoubleExpectedStateActionRewardStateActionV2 = require(Models.DeepDoubleExpectedStateActionRewardStateActionV2),
	
	MonteCarloControl = require(Models.MonteCarloControl),

	OffPolicyMonteCarloControl = require(Models.OffPolicyMonteCarloControl),
	
	REINFORCE = require(Models.REINFORCE),
	
	VanillaPolicyGradient = require(Models.VanillaPolicyGradient),
	
	ActorCritic = require(Models.ActorCritic),
	
	SoftActorCritic = require(Models.SoftActorCritic),
	
	AdvantageActorCritic = require(Models.AdvantageActorCritic),
	
	ProximalPolicyOptimization = require(Models.ProximalPolicyOptimization),
	
	ProximalPolicyOptimizationClip = require(Models.ProximalPolicyOptimizationClip),
	
	DeepDeterministicPolicyGradient = require(Models.DeepDeterministicPolicyGradient),
	
	TwinDelayedDeepDeterministicPolicyGradient = require(Models.TwinDelayedDeepDeterministicPolicyGradient),
	
	GenerativeAdversarialNetwork = require(Models.GenerativeAdversarialNetwork),
	
	ConditionalGenerativeAdversarialNetwork = require(Models.ConditionalGenerativeAdversarialNetwork),

	WassersteinGenerativeAdversarialNetwork = require(Models.WassersteinGenerativeAdversarialNetwork),

	ConditionalWassersteinGenerativeAdversarialNetwork = require(Models.ConditionalWassersteinGenerativeAdversarialNetwork),
	
}

AqwamMachineAndDeepLearningLibrary.Optimizers = {

	RootMeanSquarePropagation = require(Optimizers.RootMeanSquarePropagation),

	Momentum = require(Optimizers.Momentum),

	AdaptiveGradient = require(Optimizers.AdaptiveGradient),

	AdaptiveGradientDelta = require(Optimizers.AdaptiveGradientDelta),

	AdaptiveMomentEstimation = require(Optimizers.AdaptiveMomentEstimation),

	AdaptiveMomentEstimationMaximum = require(Optimizers.AdaptiveMomentEstimationMaximum),

	NesterovAcceleratedAdaptiveMomentEstimation = require(Optimizers.NesterovAcceleratedAdaptiveMomentEstimation),

	Gravity = require(Optimizers.Gravity),

	LearningRateStepDecay = require(Optimizers.LearningRateStepDecay),

	LearningRateTimeDecay = require(Optimizers.LearningRateTimeDecay),

}

AqwamMachineAndDeepLearningLibrary.ValueSchedulers = {

	TimeDecay = require(ValueSchedulers.TimeDecay),

	StepDecay = require(ValueSchedulers.StepDecay)

}

AqwamMachineAndDeepLearningLibrary.ExperienceReplays = {

	UniformExperienceReplay = require(ExperienceReplays.UniformExperienceReplay),

	PrioritizedExperienceReplay = require(ExperienceReplays.PrioritizedExperienceReplay),

	NStepExperienceReplay = require(ExperienceReplays.NStepExperienceReplay),

}

AqwamMachineAndDeepLearningLibrary.QuickSetups = {

	CategoricalPolicy = require(QuickSetups.CategoricalPolicy),

	DiagonalGaussianPolicy = require(QuickSetups.DiagonalGaussianPolicy)

}

AqwamMachineAndDeepLearningLibrary.ReinforcementLearningStrategies = {
	
	RandomNetworkDistillation = require(ReinforcementLearningStrategies.RandomNetworkDistillation),
	
	GenerativeAdversarialImitationLearning = require(ReinforcementLearningStrategies.GenerativeAdversarialImitationLearning),

	WassersteinGenerativeAdversarialImitationLearning = require(ReinforcementLearningStrategies.WassersteinGenerativeAdversarialImitationLearning),
	
}

AqwamMachineAndDeepLearningLibrary.DistributedTrainingStrategies = {
	
	DistributedGradientsCoordinator = require(DistributedTrainingStrategies.DistributedGradientsCoordinator),

	DistributedModelParametersCoordinator = require(DistributedTrainingStrategies.DistributedModelParametersCoordinator),
	
}

AqwamMachineAndDeepLearningLibrary.Others = {
	
	Regularizer = require(Others.Regularizer),
	
	TrainingModifier = require(Others.TrainingModifier),
	
	ModelParametersMerger = require(Others.ModelParametersMerger),

	ModelDatasetCreator = require(Others.ModelDatasetCreator),

	ModelChecker = require(Others.ModelChecker),
	
	OneVsAll = require(Others.OneVsAll),
	
	ConfusionMatrixCreator = require(Others.ConfusionMatrixCreator),
	
	Tokenizer = require(Others.Tokenizer),
	
	StringSplitter = require(Others.StringSplitter),

	OnlineLearning = require(Others.OnlineLearning),

}

return AqwamMachineAndDeepLearningLibrary