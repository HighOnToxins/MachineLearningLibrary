namespace MachineLearningLibrary;

//TODO: Make ILayer generic with an TData.
public interface IAgent: IDifferentiable<IImage<float>, IImage<float>> 
{
    public int VariableCount();

    public void AddAll(IReadOnlyList<float> values);
    
    public void WriteToFile(BinaryWriter binWriter);

    
    public void SaveToFile(string path)
    {
        using BinaryWriter binWriter = new(File.Create(path));
        WriteToFile(binWriter);
    }

}

