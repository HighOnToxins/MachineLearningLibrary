
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
        float gradientLengthSquared = 0;

        batchSize = batchSize >= 0 ? batchSize : trainingData.Count;
        List<IReadOnlyList<float>> tempTrainingData = trainingData.ToList();

        for(int b = 0; b < batchSize && tempTrainingData.Count > 0; b++)
        {
            for(int agentIndex = 0; agentIndex < agent.VariableLength(); agentIndex++)
            {
                IReadOnlyList<float> agentGradient = agent.ComputeGradient(agentIndex, tempTrainingData[b]);
                float lossGradient = lossFunction.RunDerivative(agentGradient);

                //TODO: Try saving all gradients and then adding them to the agent instead.
                agent.AddValueAt(agentIndex, lossGradient * gradientFactor);
                gradientLengthSquared += lossGradient * lossGradient;
            }

            tempTrainingData.RemoveAt(b);
        }

        return (float)Math.Sqrt(gradientLengthSquared);
    }

    //Returns the average loss gradient.
    public float Test(Agent agent, int batchSize = -1)
    {
        batchSize = batchSize >= 0 ? batchSize : trainingData.Count;

        List<IReadOnlyList<float>> tempTrainingData = trainingData.ToList();

        float average = 0;
        for(int b = 0; b < batchSize && tempTrainingData.Count > 0; b++)
        {
            IReadOnlyList<float> prediction = agent.Run(testingData[b]);
            float lossGradient = lossFunction.RunFunc(prediction);

            average += (lossGradient - average) / (b + 1);
        }

        return average;
    }

}
