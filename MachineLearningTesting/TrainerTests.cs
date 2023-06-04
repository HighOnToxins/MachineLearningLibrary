
using MachineLearningLibrary;
using System.Globalization;

namespace MachineLearningTesting;

internal class TrainerTests
{

    private sealed class TestLossFunction: IDifferentiable<IReadOnlyList<float>, float>
    {
        public void Invoke(in IReadOnlyList<float> value, out float valueResult)
        {
            valueResult = value[0];
        }

        public void Invoke(
            in IReadOnlyList<float> value, 
            in IReadOnlyList<float>? gradient, 
            out float valueResult, 
            out float derivativeResult, 
            int varIndex = -1)
        {
            IReadOnlyList<float> tempGradient = gradient ?? new float[1];

            valueResult = value[0];
            derivativeResult = tempGradient[0];
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
        AgentComposite agent = new(new AffineAgent(weights));

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
        AgentComposite agent = new(new AffineAgent(weights));

        float averageLoss = trainer.Test(agent);
        float expectedAverageLoss = 13.5f;

        Assert.That(averageLoss, Is.EqualTo(expectedAverageLoss));
    }

}
