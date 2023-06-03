
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

        float gradientLengthSquared = 0;
        float[] gradient = new float[agent.VariableLength()];

        for(int b = 0; b < batchSize; b++)
        {
            int randomIndex = random.Next(trainingData.Count - 1);

            //TODO: Compute gradients multi-threaded.
            for(int agentIndex = 0; agentIndex < gradient.Length; agentIndex++)
            {
                IReadOnlyList<float> data = tempTrainingData[randomIndex];
                IReadOnlyList<float> agentGradient = agent.ComputeGradient(agentIndex, ref data);
                float lossGradient = lossFunction.InvokeDerivative(agentGradient, data);
                gradient[agentIndex] += lossGradient; 
                gradientLengthSquared += lossGradient*lossGradient;
            }

            tempTrainingData.RemoveAt(randomIndex);
        }

        float gradientLength = (float)Math.Sqrt(gradientLengthSquared);

        for(int agentIndex = 0; agentIndex < gradient.Length; agentIndex++)
        {
            gradient[agentIndex] *= gradientFactor/gradientLength;
        }

        agent.AddGradient(gradient);
        return gradientLength;
    }

    //Returns the average loss value.
    public float Test(Agent agent, int batchSize = -1)
    {
        Random random = new(); //TODO: add global random.

        batchSize = 0 <= batchSize && batchSize < testingData.Count ? batchSize : testingData.Count;
        List<IReadOnlyList<float>> tempTestingData = testingData.ToList();

        float average = 0;
        for(int b = 0; b < batchSize; b++)
        {
            int randomIndex = random.Next(tempTestingData.Count - 1);

            IReadOnlyList<float> prediction = agent.Run(testingData[randomIndex]);
            float loss = lossFunction.Invoke(prediction);

            average += (loss - average) / (b + 1); 
            tempTestingData.RemoveAt(randomIndex);
        }

        return average;
    }

}
