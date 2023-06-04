
namespace MachineLearningLibrary;

public interface IDifferentiable<T, TResult>
{

    public void Invoke(in T value, out TResult result);

    public void Invoke(
        in T value,
        in T? gradient,
        out TResult valueResult,
        out TResult derivativeResult,
        int varIndex = -1);
}
