using System.Formats.Asn1;

namespace MachineLearningLibrary;

public sealed class AgentComposite: IAgent 
{
    private readonly IReadOnlyList<IAgent> layers;

    public AgentComposite(params IAgent[] layers) {
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

    public void Invoke(in IImage<float> value, out IImage<float> result)
    {
        result = value;
        for(int i = 0; i < layers.Count; i++)
        {
            layers[i].Invoke(in result, out result);
        }
    }

    public void Invoke(
        in IImage<float> value, 
        in IImage<float>? gradient, 
        out IImage<float> valueResult, 
        out IImage<float> derivativeResult,
        int varIndex = -1)
    {
        layers[0].Invoke(value, gradient, out IImage<float> tempVal, out IImage<float> tempGradient, varIndex: varIndex);

        for(int i = 1; i < layers.Count; i++)
        {
            varIndex -= layers[i - 1].VariableCount();
            layers[i].Invoke(tempVal, tempGradient, out tempVal, out tempGradient, varIndex: varIndex);
        }

        valueResult = tempVal;
        derivativeResult = tempGradient;
    }

    public void WriteToFile(BinaryWriter binWriter)
    {
        binWriter.Write(layers.Count);

        for(int i = 0; i < layers.Count; i++)
        {
            switch(layers[i])
            {
                case AgentComposite: binWriter.Write(0); break;
                case AffineAgent: binWriter.Write(1); break;
                case ConvolutionAgent: binWriter.Write(2); break;
                default:  throw new IOException();
            }

            layers[i].WriteToFile(binWriter);
        }
    }
    
    public static AgentComposite LoadFromFile(string path)
    {
        using BinaryReader binReader = new(File.OpenRead(path));
        return ReadFromFile(binReader);
    }

    public static AgentComposite ReadFromFile(BinaryReader binReader)
    {
        int layerCount = binReader.ReadInt32();
        IAgent[] layers = new IAgent[layerCount];

        for(int i = 0; i < layerCount; i++)
        {
            int layerType = binReader.ReadInt32();

            //TODO: Change to use reflection instead of swtich, for determining layer-type.
            IAgent layer = layerType switch
            {
                0 => ReadFromFile(binReader),
                1 => AffineAgent.ReadFromFile(binReader),
                2 => ConvolutionAgent.ReadFromFile(binReader),
                _ => throw new IOException(),
            };

            layers[i] = layer;
        }

        return new AgentComposite(layers);
    }

}