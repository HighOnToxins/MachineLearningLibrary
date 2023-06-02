using MachineLearningLibrary;
using MachineLearningLibrary.Layers;
using System.Data.Common;

namespace MachineLearningTesting;

internal class LayerTests
{
    private class TestLayer: ILayer
    {
        private float weight;

        public TestLayer(float weight)
        {
            this.weight = weight;
        }

        public int VariableLength => 0;

        public int InputSize => 2;

        public int OutputSize => 2;

        public void AddValueAt(int index, float value)
        {
            if(index == 0)
            {
                weight += value;
            }
            else
            {
                throw new IndexOutOfRangeException();
            }
        }

        public IReadOnlyList<float> ForwardPass(IReadOnlyList<float> data)
        {
            return new float[] { data[0] + weight };
        }

        public void ComputeGradient(int index, ref IReadOnlyList<float> gradient, ref IReadOnlyList<float> data)
        {
            if(index == 0)
            {
                gradient = new float[] { data[0] };
            }
            else
            {
                gradient = new float[] { 0f };
            }

            data = ForwardPass(data);
        }
    }

    [Test]
    public void TestLayerComputesDataCorrectly()
    {
        TestLayer layer = new(2);

        float[] data = new float[]
        {
            1
        };

        float[] expected = new float[]
        {
            3
        };

        IReadOnlyList<float> result = layer.ForwardPass(data);

        Assert.That(result, Has.Count.EqualTo(expected.Length));
        for(int i = 0; i < result.Count; i++)
        {
            Assert.That(result, Has.ItemAt(i).EqualTo(expected[i]));
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

        AffineLayer[] layers = new AffineLayer[weights.Length];
        for(int i = 0; i < weights.Length; i++)
        {
            layers[i] = new AffineLayer(weights[i]);
        }

        Agent agent = new(layers);

        float[] data = new float[]
        {
            1, 2
        };

        float[] expected = new float[]
        {
            2318, 2802
        };

        IReadOnlyList<float> result = agent.Run(data);

        Assert.That(result, Has.Count.EqualTo(expected.Length));
        for(int i = 0; i < result.Count; i++)
        {
            Assert.That(result, Has.ItemAt(i).EqualTo(expected[i]));
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

        AffineLayer[] layers = new AffineLayer[weights.Length];
        for(int i = 0; i < weights.Length; i++)
        {
            layers[i] = new AffineLayer(weights[i]);
        }

        Agent agent = new(layers);

        float[] data = new float[]
        {
            0, 0, 0, 0
        };

        Assert.Throws<ArgumentException>(() => agent.Run(data));
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

        AffineLayer[] layers = new AffineLayer[weights.Length];
        for(int i = 0; i < weights.Length; i++)
        {
            layers[i] = new AffineLayer(weights[i]);
        }

        Agent agent = new(layers);

        float[] data = new float[]
        {
            1, 2
        };

        float[] expected = new float[]
        {
            108, 132
        };

        IReadOnlyList<float> result = agent.ComputeGradient(7, data);

        Assert.That(result, Has.Count.EqualTo(expected.Length));
        for(int i = 0; i < result.Count; i++)
        {
            Assert.That(result, Has.ItemAt(i).EqualTo(expected[i]));
        }
    }
}