
using System.Diagnostics.CodeAnalysis;

namespace MachineLearningLibrary;

public interface IImage<T>
{
    public int Rank { get; }

    public int GetLength(int dimension);

    public T GetElementAt(params int[] indecies);

    public bool TryGetElementAt(int[] indecies, [NotNullWhen(true)] out T? element);

    public void ForEach(Action<int[], T> action);

}

