
namespace MachineLearningLibrary;

public interface IDerivedFunc<T, TResult>
{
    public TResult Invoke(T input);

    //d/dx f(g(x)) = f'(g(x)) * (d/dx g(x))
    public TResult InvokeDerivative(T difInput, T input);

    //TODO: Make RunDerivative more general or make it an abstract class with other functions, to describing what to derive for.
    //TODO: Consider merging the two functions of IDerivedFunc.
}

