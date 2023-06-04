
using System;

namespace MachineLearningLibrary;

public sealed class Trainer
{
    private readonly IReadOnlyList<IReadOnlyList<float>> trainingData;
    private readonly IReadOnlyList<IReadOnlyList<float>> testingData;

    private readonly IDerivedFunc<IReadOnlyList<float>, float> lossFunction;

    public Trainer(IReadOnlyList<IReadOnlyList<float>> trainingData, 
        IReadOnlyList<IReadOnlyList<float>> testingData,
        IDerivedFunc<IReadOnlyList<float>, float> lossFunction)
    {
        this.trainingData = trainingData;
        this.testingData = testingData;
        this.lossFunction = lossFunction;
    }

    //TODO: Try running everything with checked on, making sure that no overflows happen.
    //Returns the gradient length.
    public float Train(Agent agent, float gradientFactor, int batchSize = -1)
    {
        Random random = new();
        batchSize = 0 <= batchSize && batchSize < trainingData.Count ? batchSize : trainingData.Count;
        List<IReadOnlyList<float>> tempTrainingData = trainingData.ToList();

        float[] averageGradient = new float[agent.VariableLength()];

        for(int b = 0; b < batchSize; b++)
        {
            int randomIndex = random.Next(tempTrainingData.Count);

            //TODO: Compute gradients multi-threaded.
            for(int agentIndex = 0; agentIndex < averageGradient.Length; agentIndex++)
            {
                IReadOnlyList<float> data = tempTrainingData[randomIndex];
                IReadOnlyList<float> agentGradient = agent.ComputeGradient(agentIndex, ref data);
                float lossGradient = lossFunction.InvokeDerivative(agentGradient, data);
                averageGradient[agentIndex] += (lossGradient - averageGradient[agentIndex]) / (b + 1); 
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
            //TODO: Take an option in the constructor, to determine if we should maximize or minimize.
            averageGradient[agentIndex] *= -gradientFactor/gradientLength;
        }

        agent.AddGradient(averageGradient);
        return gradientLength;
    }

    //Returns the average loss value.
    public float Test(Agent agent, int batchSize = -1)
    {
        Random random = new(); //TODO: add global random.

        batchSize = 0 <= batchSize && batchSize < testingData.Count ? batchSize : testingData.Count;
        List<IReadOnlyList<float>> tempTestingData = testingData.ToList();

        float averageLoss = 0;
        for(int b = 0; b < batchSize; b++)
        {
            int randomIndex = random.Next(tempTestingData.Count);

            IReadOnlyList<float> prediction = agent.Run(tempTestingData[randomIndex]);
            float loss = lossFunction.Invoke(prediction);

            averageLoss += (loss - averageLoss) / (b + 1); 
            tempTestingData.RemoveAt(randomIndex);
        }

        return averageLoss;
    }

}
