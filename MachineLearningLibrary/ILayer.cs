namespace MachineLearningLibrary;

//TODO: Make ILayer generic with an TData.
public interface ILayer : IDifferentiable<IReadOnlyList<float>, IReadOnlyList<float>>
{
    public int VariableCount();

    public void AddAll(IReadOnlyList<float> values);
    
    public void WriteToFile(BinaryWriter binWriter);
}

