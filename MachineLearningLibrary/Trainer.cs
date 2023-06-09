﻿
namespace MachineLearningLibrary;

public enum TrainOption{
    Maximize = 1,
    Minimize = -1
}

public sealed class Trainer
{
    private readonly IReadOnlyList<IImage<float>> trainingData;
    private readonly IReadOnlyList<IImage<float>> testingData;

    private readonly IDifferentiable<IImage<float>, float> lossFunction;

    private readonly Random random;

    public Trainer(IReadOnlyList<IImage<float>> trainingData,
        IReadOnlyList<IImage<float>> testingData,
        IDifferentiable<IImage<float>, float> lossFunction)
    {
        this.trainingData = trainingData;
        this.testingData = testingData;
        this.lossFunction = lossFunction;
        random = new Random();
    }

    //TODO: Try running everything with checked on, making sure that no overflows happen.
    //Returns the gradient length.
    public float Train(IAgent agent, float gradientFactor, int batchSize = -1, TrainOption option = TrainOption.Minimize)
    {
        batchSize = 0 <= batchSize && batchSize < trainingData.Count ? batchSize : trainingData.Count;
        List<IImage<float>> tempTrainingData = trainingData.ToList();

        float[] averageGradient = new float[agent.VariableCount()];

        for(int b = 0; b < batchSize; b++)
        {
            int randomIndex = random.Next(tempTrainingData.Count);

            //TODO: Compute gradients multi-threaded.
            for(int varIndex = 0; varIndex < averageGradient.Length; varIndex++)
            {
                agent.Invoke(tempTrainingData[randomIndex], default, out IImage<float> valueResult, out IImage<float> agentGradient, varIndex: varIndex);
                lossFunction.Invoke(valueResult, agentGradient, out _, out float lossGradient);

                averageGradient[varIndex] += (lossGradient - averageGradient[varIndex]) / (b + 1); 
            }

            tempTrainingData.RemoveAt(randomIndex);
        }

        //Compute gradient length squared.
        float gradientLengthSquared = 0;
        for(int agentIndex = 0; agentIndex < averageGradient.Length; agentIndex++)
        {
            gradientLengthSquared += averageGradient[agentIndex]*averageGradient[agentIndex];
        }

        float gradientLength = (float)Math.Sqrt(gradientLengthSquared);

        //Set the length of the gradient, and inverting.
        for(int agentIndex = 0; agentIndex < averageGradient.Length; agentIndex++)
        {
            int sign = (int)TrainOption.Minimize;
            averageGradient[agentIndex] *= sign * gradientFactor / gradientLength;
        }

        agent.AddAll(averageGradient);
        return gradientLength;
    }

    //Returns the average loss value.
    public float Test(IAgent agent, int batchSize = -1)
    {
        batchSize = 0 <= batchSize && batchSize < testingData.Count ? batchSize : testingData.Count;
        List<IImage<float>> tempTestingData = testingData.ToList();

        float averageLoss = 0;
        for(int b = 0; b < batchSize; b++)
        {
            int randomIndex = random.Next(tempTestingData.Count);

            agent.Invoke(tempTestingData[randomIndex], out IImage<float> prediction);
            lossFunction.Invoke(prediction, out float loss);

            averageLoss += (loss - averageLoss) / (b + 1); 
            tempTestingData.RemoveAt(randomIndex);
        }

        return averageLoss;
    }

}
