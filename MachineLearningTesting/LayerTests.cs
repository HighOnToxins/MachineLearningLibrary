using MachineLearningLibrary;
using System;
using System.Data.Common;
using System.Globalization;
using System.Reflection;
using System.Reflection.Emit;

namespace MachineLearningTesting;

internal class LayerTests
{
    private class TestLayer: IAgent
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
            out IReadOnlyList<float> valueResult)
        {
            valueResult = new float[] { value[0] + weight };
        }

        public void Invoke(
            in IReadOnlyList<float> value, 
            in IReadOnlyList<float>? derivative, 
            out IReadOnlyList<float> valueResult, 
            out IReadOnlyList<float> derivativeResult, 
            int varIndex = -1)
        {
            valueResult = new float[] { value[0] + weight };

            if(varIndex == 0)
            {
                derivativeResult = new float[] { value[0] };
            }
            else
            {
                derivativeResult = new float[] { 0f };
            }
        }

        public void WriteToFile(BinaryWriter binWriter)
        {
            throw new NotImplementedException();
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

        layer.Invoke(data, out IReadOnlyList<float> result);

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

        AffineAgent[] layers = new AffineAgent[weights.Length];
        for(int i = 0; i < weights.Length; i++)
        {
            layers[i] = new AffineAgent(weights[i]);
        }

        AgentComposite agent = new(layers);

        float[] data = new float[]
        {
            1, 2
        };

        float[] expected = new float[]
        {
            2318, 2802
        };

        agent.Invoke(data, out IReadOnlyList<float> result);

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

        AffineAgent[] layers = new AffineAgent[weights.Length];
        for(int i = 0; i < weights.Length; i++)
        {
            layers[i] = new AffineAgent(weights[i]);
        }

        AgentComposite agent = new(layers);

        float[] data = new float[]
        {
            0, 0, 0, 0
        };

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

        AgentComposite agent = new(layers);

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
            7);

        Assert.That(result, Has.Count.EqualTo(expected.Length));
        for(int i = 0; i < result.Count; i++)
        {
            Assert.That(result, Has.ItemAt(i).EqualTo(expected[i]));
        }
    }
}