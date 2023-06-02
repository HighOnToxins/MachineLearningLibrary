using MachineLearningLibrary.Layers;

namespace MachineLearningLibrary;

public sealed class Agent
{
    private readonly IReadOnlyList<ILayer> layers;

    public Agent(params ILayer[] layers) {
        this.layers = layers;

        //Check if the inputs and output sizes match.
        for(int i = 1; i < layers.Length; i++)
        {
            if(layers[i-1].OutputSize != layers[i].InputSize)
            {
                throw new ArgumentException();
            }
        }
    }

    public int VariableLength()
    {
        int sum = 0;
        for(int i = 0; i < layers.Count; i++)
        {
            sum += layers.Count;
        }
        return sum;
    }

    public int InputSize { get => layers[0].InputSize; }

    public int OutputSize { get => layers[^0].OutputSize; }

    public IReadOnlyList<float> Run(IReadOnlyList<float> data)
    {
        for(int i = 0; i < layers.Count; i++)
        {
            data = layers[i].ForwardPass(data);
        }

        return data;
    }

    //TODO: find a way to use a simpler com ComputeGradient function.
    public IReadOnlyList<float> ComputeGradient(int index, IReadOnlyList<float> data)
    {
        IReadOnlyList<float> gradient = new float[InputSize];
        layers[0].ComputeGradient(index, ref gradient, ref data);

        for(int i = 1; i < layers.Count; i++)
        {
            index -= layers[i].VariableLength;
            layers[i].ComputeGradient(index, ref gradient, ref data); 
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