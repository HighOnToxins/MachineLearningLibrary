using MachineLearningLibrary;
using System;
using System.Data.Common;
using System.Reflection;
using System.Reflection.Emit;

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
        public int VariableCount() => 0;

        public void AddAll(IReadOnlyList<float> values)
        {
            weight += values[0];
        }

        public void Invoke(
            in IReadOnlyList<float> value, 
            in IReadOnlyList<float>? gradient, 
            out IReadOnlyList<float> valueResult, 
            out IReadOnlyList<float> derivativeResult, 
            ComputeOptions options = ComputeOptions.ValueAndDerivative, 
            int varIndex = -1)
        {
            valueResult = Array.Empty<float>();
            derivativeResult = Array.Empty<float>();

            if(options.HasFlag(ComputeOptions.Value))
            {
                valueResult = new float[] { value[0] + weight };
            }

            if(!options.HasFlag(ComputeOptions.Derivative))
            {
                return;
            }

            if(varIndex == 0)
            {
                valueResult = new float[] { value[0] };
            }
            else
            {
                valueResult = new float[] { 0f };
            }
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

        layer.Invoke(
            data,
            default,
            out IReadOnlyList<float> result,
            out _,
            ComputeOptions.Value);

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

        IReadOnlyList<float> data = new float[]
        {
            1, 2
        };

        float[] expected = new float[]
        {
            108, 132
        };

        agent.Invoke(
            data,
            default,
            out _,
            out IReadOnlyList<float> result,
            ComputeOptions.Derivative,
            7);

        Assert.That(result, Has.Count.EqualTo(expected.Length));
        for(int i = 0; i < result.Count; i++)
        {
            Assert.That(result, Has.ItemAt(i).EqualTo(expected[i]));
        }
    }
}