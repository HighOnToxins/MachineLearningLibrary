using System.Formats.Asn1;

namespace MachineLearningLibrary;

public sealed class Agent: ILayer
{
    private readonly IReadOnlyList<ILayer> layers;

    public Agent(params ILayer[] layers) {
        this.layers = layers;
    }

    public int VariableCount() //Chache variable count when calculated.
    {
        int sum = 0;
        for(int i = 0; i < layers.Count; i++)
        {
            sum += layers.Count;
        }
        return sum;
    }

    public void AddAll(IReadOnlyList<float> values)
    {
        float[] vals = values.ToArray(); //TODO: Create get range method instead.
        int startIndex = 0;
        for(int i = 0; i < layers.Count; i++)
        {
            int variableCount = layers[i].VariableCount();
            layers[i].AddAll(vals[startIndex..(startIndex + variableCount - 1)]);
            startIndex += variableCount;
        }
    }

    public IReadOnlyList<float> Run(IReadOnlyList<float> value)
    {
        for(int i = 0; i < layers.Count; i++)
        {
            layers[i].Invoke(value, default, out value, out _, ComputeOptions.Value);
        }

        return value;
    }

    public void Invoke(
        in IReadOnlyList<float> value, 
        in IReadOnlyList<float>? gradient, 
        out IReadOnlyList<float> valueResult, 
        out IReadOnlyList<float> derivativeResult, 
        ComputeOptions options = ComputeOptions.ValueAndDerivative, 
        int varIndex = -1)
    {
        if(!options.HasFlag(ComputeOptions.Derivative))
        {
            valueResult = Run(value);
            derivativeResult = Array.Empty<float>();
            return;
        }

        layers[0].Invoke(value, gradient, out IReadOnlyList<float> tempVal, out IReadOnlyList<float> tempGradient, varIndex: varIndex);

        for(int i = 1; i < layers.Count; i++)
        {
            varIndex -= layers[i - 1].VariableCount();
            layers[i].Invoke(tempVal, tempGradient, out tempVal, out tempGradient, varIndex: varIndex);
        }

        valueResult = tempVal;
        derivativeResult = tempGradient;
    }

    public void SaveToFile(string path)
    {
        using BinaryWriter binWriter = new(File.Create(path));
        WriteToFile(binWriter);
    }

    public void WriteToFile(BinaryWriter binWriter)
    {
        binWriter.Write(layers.Count);

        for(int i = 0; i < layers.Count; i++)
        {
            switch(layers[i])
            {
                case Agent: binWriter.Write(0); break;
                case AffineLayer: binWriter.Write(1); break;
                default:  throw new IOException();
            }

            layers[i].WriteToFile(binWriter);
        }
    }

    public static Agent LoadFromFile(string path)
    {
        using BinaryReader binReader = new(File.OpenRead(path));
        return ReadFromFile(binReader);
    }

    public static Agent ReadFromFile(BinaryReader binReader)
    {
        int layerCount = binReader.ReadInt32();
        ILayer[] layers = new ILayer[layerCount];

        for(int i = 0; i < layerCount; i++)
        {
            int layerType = binReader.ReadInt32();

            //TODO: Change to use reflection instead of swtich, for determining layer-type.
            ILayer layer = layerType switch
            {
                0 => ReadFromFile(binReader),
                1 => AffineLayer.ReadFromFile(binReader),
                _ => throw new IOException(),
            };
            layers[i] = layer;
        }

        return new Agent(layers);
    }

}