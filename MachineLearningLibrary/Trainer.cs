
namespace MachineLearningLibrary;

public enum TrainOption{
    Maximize = 1,
    Minimize = -1 
}

//TODO: Consider using a generic for the answer type rather than an integer.
public sealed class Trainer
{
    private readonly IReadOnlyList<IReadOnlyList<float>> trainingData;
    private readonly IReadOnlyList<int> trainingAnswers;

    private readonly IReadOnlyList<IReadOnlyList<float>> testingData;
    public readonly IReadOnlyList<int> testingAnswers;

    private readonly ILossFunc lossFunction;

    private readonly Random random;

    public Trainer(IReadOnlyList<IReadOnlyList<float>> trainingData,
        IReadOnlyList<int> trainingAnswers,
        IReadOnlyList<IReadOnlyList<float>> testingData,
        IReadOnlyList<int> testingAnswers,
        ILossFunc lossFunction)
    {
        this.trainingData = trainingData;
        this.trainingAnswers = trainingAnswers;
        this.testingData = testingData;
        this.testingAnswers = testingAnswers;
        this.lossFunction = lossFunction;
        random = new Random();
    }

    //TODO: Try running everything with checked on, making sure that no overflows happen.
    //Returns the gradient length.
    public float Train(IAgent agent, float gradientSpeed, int batchSize = -1, TrainOption option = TrainOption.Minimize)
    {
        batchSize = 0 <= batchSize && batchSize < trainingData.Count ? batchSize : trainingData.Count;
        List<IReadOnlyList<float>> tempTrainingData = trainingData.ToList();
        List<int> tempTrainingAnswers  = trainingAnswers.ToList();

        float[] averageGradient = new float[agent.VariableCount()];

        for(int b = 0; b < batchSize; b++)
        {
            int randomIndex = random.Next(tempTrainingData.Count);

            //TODO: Compute gradients multi-threaded.
            for(int varIndex = 0; varIndex < averageGradient.Length; varIndex++)
            {
                agent.Invoke(tempTrainingData[randomIndex], default, out IReadOnlyList<float> valueResult, out IReadOnlyList<float> agentGradient, varIndex: varIndex);
                lossFunction.Answer = tempTrainingAnswers[randomIndex];
                lossFunction.Invoke(valueResult, agentGradient, out _, out float lossGradient);

                averageGradient[varIndex] += (lossGradient - averageGradient[varIndex]) / (b + 1); 
            }

            tempTrainingData.RemoveAt(randomIndex);
            tempTrainingAnswers.RemoveAt(randomIndex);
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
            int sign = (int)option;
            averageGradient[agentIndex] *= sign * gradientSpeed/gradientLength;
        }

        agent.AddAll(averageGradient);
        return gradientLength;
    }

    //Returns the average loss value.
    public float Test(IAgent agent, int batchSize = -1)
    {
        batchSize = 0 <= batchSize && batchSize < testingData.Count ? batchSize : testingData.Count;
        List<IReadOnlyList<float>> tempTestingData = testingData.ToList();
        List<int> tempTestingAnswers = testingAnswers.ToList();

        float averageLoss = 0;
        for(int b = 0; b < batchSize; b++)
        {
            int randomIndex = random.Next(tempTestingData.Count);

            agent.Invoke(tempTestingData[randomIndex], out IReadOnlyList<float> prediction);
            lossFunction.Answer = tempTestingAnswers[randomIndex];
            lossFunction.Invoke(prediction, out float loss);

            averageLoss += (loss - averageLoss) / (b + 1); 
            tempTestingData.RemoveAt(randomIndex);
            tempTestingAnswers.RemoveAt(randomIndex);
        }

        return averageLoss;
    }

}
