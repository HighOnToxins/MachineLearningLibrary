
namespace MachineLearningLibrary;

public sealed class Trainer
{
    private readonly IReadOnlyList<IReadOnlyList<float>> trainingData;
    private readonly IReadOnlyList<IReadOnlyList<float>> testingData;

    private readonly Func<IReadOnlyList<float>, float> lossFunction;
    private readonly Func<IReadOnlyList<float>, float> lossDerivative;

    public Trainer(IReadOnlyList<IReadOnlyList<float>> trainingData, 
        IReadOnlyList<IReadOnlyList<float>> testingData,
        Func<IReadOnlyList<float>, float> lossFunction,
        Func<IReadOnlyList<float>, float> lossDerivative)
    {
        this.trainingData = trainingData;
        this.testingData = testingData;
        this.lossFunction = lossFunction;
        this.lossDerivative = lossDerivative;
    }

    public void Train(Agent agent, int batchSize = -1)
    {



        throw new NotImplementedException();
    }

    public double Test(Agent agent, int batchSize = -1)
    {


        throw new NotImplementedException();
    }

}
