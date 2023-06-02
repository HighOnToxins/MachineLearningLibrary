
using System;

namespace MachineLearningLibrary;


public interface IDerivedFunc<T, TResult>
{
    public TResult RunFunc(T input);
    public TResult RunDerivative(T input);
}

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

    //Returns the gradient length.
    public float Train(Agent agent, float gradientFactor, int batchSize = -1)
    {
        Random random = new();
        batchSize = batchSize >= 0 ? Math.Min(batchSize, trainingData.Count) : trainingData.Count;
        List<IReadOnlyList<float>> tempTrainingData = trainingData.ToList();

        float gradientLengthSquared = 0;
        float[] gradientAverage = new float[agent.VariableLength()];

        for(int b = 0; b < batchSize; b++)
        {
            int randomIndex = random.Next(trainingData.Count - 1);

            //TODO: Compute gradients multi-threaded.
            for(int agentIndex = 0; agentIndex < gradientAverage.Length; agentIndex++)
            {
                IReadOnlyList<float> agentGradient = agent.ComputeGradient(agentIndex, tempTrainingData[randomIndex]);
                float lossGradient = lossFunction.RunDerivative(agentGradient);

                gradientAverage[agentIndex] += (lossGradient*gradientFactor - gradientAverage[agentIndex]) / (b + 1);
                gradientLengthSquared += lossGradient*lossGradient;
            }

            tempTrainingData.RemoveAt(randomIndex);
        }

        agent.AddGradient(gradientAverage);
        return (float)Math.Sqrt(gradientLengthSquared);
    }

    //Returns the average loss value.
    public float Test(Agent agent, int batchSize = -1)
    {
        Random random = new();

        batchSize = 0 <= batchSize ? Math.Min(batchSize, testingData.Count) : testingData.Count;
        List<IReadOnlyList<float>> tempTestingData = testingData.ToList();

        float average = 0;
        for(int b = 0; b < batchSize; b++)
        {
            int randomIndex = random.Next(tempTestingData.Count - 1);

            IReadOnlyList<float> prediction = agent.Run(testingData[randomIndex]);
            float loss = lossFunction.RunFunc(prediction);

            average += (loss - average) / (b + 1); 
            tempTestingData.RemoveAt(randomIndex);
        }

        return average;
    }

}
