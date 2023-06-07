using System.Reflection;

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
        WriteAnyToFile(agent, binWriter);
    }

    public static IAgent LoadFromFile(string path)
    {
        using BinaryReader binReader = new(File.OpenRead(path));
        return ReadAnyFromFile(binReader);
    }

    public static void WriteAnyToFile(IAgent agent, BinaryWriter binWriter)
    {
        binWriter.Write(agent.GetType().Name.GetHashCode());
        agent.WriteToFile(binWriter);
    }

    public static IAgent ReadAnyFromFile(BinaryReader binReader)
    {
        int agentCode = binReader.ReadInt32();
        Type agentType = IOAgents.First(t => t.Name.GetHashCode() == agentCode);
        MethodInfo? method = agentType.GetMethod("ReadFromFile",
            BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy);

        if(method is null || method.Invoke(null, new object[] { binReader }) is not IAgent agent)
        {
            throw new IOException();
        }

        return agent;
    }

    static IAgent()
    {
        IOAgents = Assembly.GetExecutingAssembly()
            .GetTypes()
            .Where(t => t.IsAssignableTo(typeof(IAgent)) && t != typeof(IAgent))
            .ToArray();
    }

    public static readonly Type[] IOAgents;
}
