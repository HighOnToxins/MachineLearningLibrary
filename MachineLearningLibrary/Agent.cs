using MachineLearningLibrary.Layers;

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

    //TODO: find a way to use a simpler com ComputeGradient function.
    internal IReadOnlyList<float> ComputeGradient(int index, IReadOnlyList<float> data)
    {
        IReadOnlyList<float> gradient = Array.Empty<float>();
        layers[0].ComputeGradient(index, ref gradient, ref data);

        for(int i = 1; 0 < layers.Count; i++)
        {
            index -= layers[i].VariableLength;
            layers[0].ComputeGradient(index, ref gradient, ref data); 
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