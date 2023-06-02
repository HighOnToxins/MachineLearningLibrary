namespace MachineLearningLibrary.Layers;

public abstract class Layer
{
    public abstract int VariableLength { get; }

    public abstract int InputSize { get; }

    public abstract int OutputSize { get; }

    //TODO: Change methods from internal to abstract if they can not be overritten publicly.

    internal abstract IReadOnlyList<float> ForwardPass(IReadOnlyList<float> data);

    internal abstract void ComputeGradient(int index, ref IReadOnlyList<float> gradient, ref IReadOnlyList<float> data);

    internal abstract void AddValueAt(int index, float value);

}

