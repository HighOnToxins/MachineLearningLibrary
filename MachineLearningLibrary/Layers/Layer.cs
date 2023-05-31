namespace MachineLearningLibrary.Layers;

public abstract class Layer
{
    public abstract int VariableLength { get; }

    public abstract int InputSize { get; }

    public abstract int OutputSize { get; }

    internal abstract IReadOnlyList<float> ForwardPass(IReadOnlyList<float> data);

    internal abstract IReadOnlyList<float> ComputeGradient(int index, IReadOnlyList<float> data);

    internal abstract IReadOnlyList<float> ComputeGradientFromGradient(int index, IReadOnlyList<float> gradient, IReadOnlyList<float> data);

    internal abstract void AddValueAt(int index, float value);

}

