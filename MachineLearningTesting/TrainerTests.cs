
using MachineLearningLibrary;
using MachineLearningLibrary.Layers;

namespace MachineLearningTesting;

internal class TrainerTests
{

    private sealed class TestLossFunction: IDerivedFunc<IReadOnlyList<float>, float>
    {
        public float Invoke(IReadOnlyList<float> input)
        {
            return input[0];
        }

        public float InvokeDerivative(IReadOnlyList<float> difInput, IReadOnlyList<float> input)
        {
            return difInput[0];
        }
    }

    [Test]
    public void TrainerComputesGradientLengthCorrectly()
    {
        IReadOnlyList<IReadOnlyList<float>> trainingData = new IReadOnlyList<float>[] {
            new float[]{1},
            new float[]{2},
            new float[]{3},
            new float[]{4},
        };
        Trainer trainer = new(trainingData, null!, new TestLossFunction());

        float[,] weights = new float[,]{
            {5f, 1f},
        };
        Agent agent = new(new AffineLayer(weights));

        float gradientLength = trainer.Train(agent, 1);
        float expectedGradientLength = 2.5f;

        Assert.That(gradientLength, Is.EqualTo(expectedGradientLength));
    }

    [Test]
    public void TrainerComputesAverageLossCorrectly()
    {
        IReadOnlyList<IReadOnlyList<float>> testData = new IReadOnlyList<float>[] {
            new float[]{1},
            new float[]{2},
            new float[]{3},
            new float[]{4},
        };

        Trainer trainer = new(null!, testData, new TestLossFunction());

        float[,] weights = new float[,]{
            {5f, 1f},
        };
        Agent agent = new(new AffineLayer(weights));

        float averageLoss = trainer.Test(agent);
        float expectedAverageLoss = 13.5f;

        Assert.That(averageLoss, Is.EqualTo(expectedAverageLoss));
    }

}
