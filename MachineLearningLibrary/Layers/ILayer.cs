namespace MachineLearningLibrary.Layers;

public interface ILayer
{
    public int VariableLength { get; }

    public int InputSize { get; }

    public int OutputSize { get; }

    public IReadOnlyList<float> ForwardPass(IReadOnlyList<float> data);

    //Computes the gradient based on the data, and the gradient of that data.
    public void ComputeGradient(int index, ref IReadOnlyList<float> gradient, ref IReadOnlyList<float> data);

    public void AddValueAt(int index, float value);

}

