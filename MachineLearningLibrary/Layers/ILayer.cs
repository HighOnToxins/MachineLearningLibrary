namespace MachineLearningLibrary.Layers;

//TODO: Make layer a derived func.
public interface ILayer
{
    public int VariableLength { get; }

    public int InputSize { get; }

    public int OutputSize { get; }

    public void AddGradient(float[] floats);

    //TODO: Change from IReadOnlyList to some other generic array/list. (make it easy to convert to from other data types/structures)
    public IReadOnlyList<float> ForwardPass(IReadOnlyList<float> data); 

    //Computes the gradient based on the data, and the gradient of that data.
    public void ComputeGradient(int index, ref IReadOnlyList<float> gradient, ref IReadOnlyList<float> data);

}

