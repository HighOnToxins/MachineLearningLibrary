namespace MachineLearningLibrary;

//TODO: Make ILayer generic with an TData.
public interface IAgent: IDifferentiable<IReadOnlyList<float>, IReadOnlyList<float>> 
{
    public int VariableCount();

    public void AddAll(IReadOnlyList<float> values);
    
    public void WriteToFile(BinaryWriter binWriter);

    public static void SaveToFile(IAgent agent, string path)
    {
        using BinaryWriter binWriter = new(File.Create(path));
        switch(agent)
        {
            case AgentComposite: binWriter.Write(0); break;
            case AffineAgent: binWriter.Write(1); break;
            case Convolution2DAgent: binWriter.Write(2); break;
            default: throw new IOException();
        }
        agent.WriteToFile(binWriter);
    }

    public static IAgent LoadFromFile(string path)
    {
        using BinaryReader binReader = new(File.OpenRead(path));

        int agentType = binReader.ReadInt32();
        return agentType switch
        {
            0 => AgentComposite.ReadFromFile(binReader),
            1 => AffineAgent.ReadFromFile(binReader),
            2 => Convolution2DAgent.ReadFromFile(binReader),
            _ => throw new IOException(),
        };
    }
}

