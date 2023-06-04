namespace MachineLearningLibrary;

//TODO: Make ILayer generic with an TData.
public interface IAgent: IDifferentiable<IReadOnlyList<float>, IReadOnlyList<float>> 
{
    public int VariableCount();

    public void AddAll(IReadOnlyList<float> values);
    
    public void WriteToFile(BinaryWriter binWriter);
}

