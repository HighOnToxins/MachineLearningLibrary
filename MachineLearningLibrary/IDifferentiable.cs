
namespace MachineLearningLibrary;

public enum ComputeOptions
{
    Derivative = 0x01,
    Value = 0x10,
    ValueAndDerivative = Derivative | Value,
}

public interface IDifferentiable<T, TResult>
{
    public void Invoke(
        in T value,
        in T? gradient,
        out TResult valueResult,
        out TResult derivativeResult,
        ComputeOptions options = ComputeOptions.ValueAndDerivative,
        int varIndex = -1);
}
