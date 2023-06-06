
using MachineLearningLibrary;
using System.Globalization;

namespace MachineLearningTesting;

internal class TrainerTests
{

    private sealed class TestLossFunction: IDifferentiable<IImage<float>, float>
    {
        public void Invoke(in IImage<float> value, out float valueResult)
        {
            valueResult = value.GetElementAt(0);
        }

        public void Invoke(
            in IImage<float> value, 
            in IImage<float>? gradient, 
            out float valueResult, 
            out float derivativeResult, 
            int varIndex = -1)
        {
            IImage<float> tempGradient = gradient ?? new ArrayImage<float>(1);

            valueResult = value.GetElementAt(0);
            derivativeResult = tempGradient.GetElementAt(0);
        }
    }

    [Test]
    public void TrainerComputesGradientLengthCorrectly()
    {
        IReadOnlyList<IImage<float>> trainingData = new IImage<float>[] {
            new ArrayImage<float>(new float[]{1}),
            new ArrayImage < float >(new float[] { 2 }),
            new ArrayImage < float >(new float[] { 3 }),
            new ArrayImage < float >(new float[] { 4 }),
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
        IReadOnlyList<IImage<float>> testingData = new IImage<float>[] {
            new ArrayImage<float>(new float[]{1}),
            new ArrayImage < float >(new float[] { 2 }),
            new ArrayImage < float >(new float[] { 3 }),
            new ArrayImage < float >(new float[] { 4 }),
        };
        Trainer trainer = new(null!, testingData, new TestLossFunction());

        float[,] weights = new float[,]{
            {5f, 1f},
        };
        AgentComposite agent = new(new AffineAgent(weights));

        float averageLoss = trainer.Test(agent);
        float expectedAverageLoss = 13.5f;

        Assert.That(averageLoss, Is.EqualTo(expectedAverageLoss));
    }

}
