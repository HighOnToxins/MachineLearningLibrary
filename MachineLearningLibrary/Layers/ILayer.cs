namespace MachineLearningLibrary.Layers;

public interface ILayer
{
    public int VariableLength { get; }

    public int InputSize { get; }

    public int OutputSize { get; }

    //TODO: Change methods from internal to abstract if they can not be overritten publicly.

    public IReadOnlyList<float> ForwardPass(IReadOnlyList<float> data);

    public void ComputeGradient(int index, ref IReadOnlyList<float> gradient, ref IReadOnlyList<float> data);

    public void AddValueAt(int index, float value);

}

