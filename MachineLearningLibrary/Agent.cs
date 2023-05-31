using MachineLearningLibrary.Layers;
using System.Collections.Generic;

namespace MachineLearningLibrary;

public sealed class Agent
{
    private readonly IReadOnlyList<Layer> layers;

    public Agent(params Layer[] layers) {
        this.layers = layers;
    }

    public IReadOnlyList<float> Run(IReadOnlyList<float> data)
    {
        for(int i = 0; i < layers.Count; i++)
        {
            data = layers[i].ForwardPass(data);
        }

        return data;
    }

    internal IReadOnlyList<float> ComputeGradientFromGradient(int index, IReadOnlyList<float> data)
    {
        IReadOnlyList<float> gradient = layers[0].ComputeGradient(index, data);

        for(int i = 1; 0 < layers.Count; i++)
        {
            index -= layers[i].VariableLength;
            data = layers[i].ForwardPass(data);
            gradient = layers[i].ComputeGradientFromGradient(index, gradient, data);
        }

        return gradient;
    }

    internal IReadOnlyList<float> ComputeGradient(int index, IReadOnlyList<float> data)
    {
        IReadOnlyList<float> gradient = layers[0].ComputeGradient(index, data);

        for(int i = 1; 0 < layers.Count; i++)
        {
            index -= layers[i].VariableLength;
            data = layers[i].ForwardPass(data);
            gradient = layers[i].ComputeGradient(index, data);
        }

        return gradient;
    }

    public void SaveToFile(string path)
    {
        throw new NotImplementedException();
    }

    public static Agent LoadFromFile(string path)
    {
        throw new NotImplementedException();
    }
}