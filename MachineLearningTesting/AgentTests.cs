using MachineLearningLibrary;

namespace MachineLearningTesting;

internal class AgentTests
{
    private class TestAgent: IAgent
    {
        private float weight;

        public TestAgent(float weight)
        {
            this.weight = weight;
        }
        public int VariableCount() => 0;

        public void AddAll(IReadOnlyList<float> values)
        {
            weight += values[0];
        }

        public void Invoke(
            in IImage<float> value,
            out IImage<float> valueResult)
        {
            ArrayImage<float> tempValueResult = new(1);
            tempValueResult.AssignElementAt(value.ElementAt(0) + weight, 0);
            valueResult = tempValueResult;
        }

        public void Invoke(
            in IImage<float> value, 
            in IImage<float>? derivative, 
            out IImage<float> valueResult, 
            out IImage<float> derivativeResult, 
            int varIndex = -1)
        {
            ArrayImage<float> tempValueResult = new (1);
            tempValueResult.AssignElementAt(value.ElementAt(0) + weight, 0);
            valueResult = tempValueResult;

            ArrayImage<float> tempDerivativeResult = new(1);
            if(varIndex == 0)
            {
                tempDerivativeResult.AssignElementAt(value.ElementAt(0), 0);
            }
            else
            {
                tempDerivativeResult.AssignElementAt(0, 0);
            }
            derivativeResult = tempDerivativeResult;
        }

        public void WriteToFile(BinaryWriter binWriter)
        {
            throw new NotImplementedException();
        }
    }

    [Test]
    public void TestLayerComputesDataCorrectly()
    {
        TestAgent agent = new(2);

        IImage<float> data = new ArrayImage<float>(new float[]
        {
            1
        });

        float[] expected = new float[]
        {
            3
        };

        agent.Invoke(data, out IImage<float> result);

        Assert.That(result.GetLength(0), Is.EqualTo(expected.Length));
        for(int i = 0; i < result.GetLength(0); i++)
        {
            Assert.That(result.ElementAt(i), Is.EqualTo(expected[i]));
        }

        Assert.Pass();
    }

    [Test]
    public void AffineLayerComputesDataCorrectly()
    {
        float[][,] weights = new float[][,]{
            new float[,]{ 
                {1f, 2f, 1f}, 
                {3f, 4f, 1f},
            },
            new float[,]{
                {5f, 6f, 1f},
                {7f, 8f, 1f},
            },
            new float[,]{
                {9f, 10f, 1f},
                {11f, 12f, 1f},
            },
        };

        AffineAgent[] layers = new AffineAgent[weights.Length];
        for(int i = 0; i < weights.Length; i++)
        {
            layers[i] = new AffineAgent(weights[i]);
        }

        IAgent agent = new AgentComposite(layers);

        IImage<float> data = new ArrayImage<float>(new float[]
        {
            1, 2
        });

        float[] expected = new float[]
        {
            2318, 2802
        };

        agent.Invoke(data, out IImage<float> result);

        Assert.That(result.GetLength(0), Is.EqualTo(expected.Length));
        for(int i = 0; i < result.GetLength(0); i++)
        {
            Assert.That(result.ElementAt(i), Is.EqualTo(expected[i]));
        }
    }

    [Test]
    public void AffineLayerDoesNotComupteDataOfWrongSize()
    {
        float[][,] weights = new float[][,]{
            new float[,]{
                {0, 0, 0},
                {0, 0, 0},
            },
        };

        AffineAgent[] layers = new AffineAgent[weights.Length];
        for(int i = 0; i < weights.Length; i++)
        {
            layers[i] = new AffineAgent(weights[i]);
        }

        IAgent agent = new AgentComposite(layers);

        IImage<float> data = new ArrayImage<float>(new float[]
        {
            0, 0, 0, 0
        });

        Assert.Throws<ArgumentException>(() => agent.Invoke(data, out _));
    }

    [Test]
    public void AffineLayerComputesGradientCorrectly()
    {
        float[][,] weights = new float[][,]{
            new float[,]{
                {1f, 2f, 1f},
                {3f, 4f, 1f},
            },
            new float[,]{
                {5f, 6f, 1f},
                {7f, 8f, 1f},
            },
            new float[,]{
                {9f, 10f, 1f},
                {11f, 12f, 1f},
            },
        };

        AffineAgent[] layers = new AffineAgent[weights.Length];
        for(int i = 0; i < weights.Length; i++)
        {
            layers[i] = new AffineAgent(weights[i]);
        }

        IAgent agent = new AgentComposite(layers);

        IImage<float> data = new ArrayImage<float>(new float[]
        {
            1, 2
        });

        float[] expected = new float[]
        {
            108, 132
        };

        agent.Invoke(data, default, out _, out IImage<float> result, 7);

        Assert.That(result.GetLength(0), Is.EqualTo(expected.Length));
        for(int i = 0; i < result.GetLength(0); i++)
        {
            Assert.That(result.ElementAt(i), Is.EqualTo(expected[i]));
        }
    }

    [Test]
    public void CovolutionAgentComputesProperly()
    {
        ArrayImage<float> array = new(3);
        array.AssignByActualIndex((i, v) => 1);

        ConvolutionAgent agent = new(
            array, 
            new int[] {5}, 
            new int[] {3});

        ArrayImage<float> image = new(5);
        image.AssignByActualIndex((i, v) => i);

        agent.Invoke(
            image, 
            out IImage<float> resultImages);

        float[] expected = new float[] { 1, 2, 3};

        for(int i = 0; i < array.GetLength(0); i++)
        {
            Assert.That(resultImages.ElementAt(i), Is.EqualTo(expected[i]));
        }
    }

    [Test]
    public void MultiDimensionalCovolutionComputesProper()
    {
        ArrayImage<float> kernals = new(2, 3, 4);
        kernals.AssignByActualIndex((i, _) => 1);

        ConvolutionAgent agent = new(
            kernals,
            new int[] { 2, 3, 4 },
            new int[] { 2, 2, 2 });

        ArrayImage<float> image = new(2, 3, 4);
        image.AssignByActualIndex((i, v) => i);

        agent.Invoke(image, out IImage<float> resultImage);

        Assert.That(resultImage.GetLength(0), Is.EqualTo(2));
        for(int a = 0; a < resultImage.GetLength(0); a++)
        {
            Assert.That(resultImage.GetLength(1), Is.EqualTo(2));
            for(int b = 0; b < resultImage.GetLength(1); b++)
            {
                Assert.That(resultImage.GetLength(2), Is.EqualTo(2));
                for(int c = 0; c < resultImage.GetLength(2); c++)
                {
                    Assert.That(
                        resultImage.ElementAt(a, b, c),
                        Is.Not.EqualTo(0)
                    );
                }
            }
        }

    }
}