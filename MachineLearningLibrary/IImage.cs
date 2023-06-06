
using System.Diagnostics.CodeAnalysis;

namespace MachineLearningLibrary;

public interface IImage<T>
{
    public int Rank { get; }

    public int ElementCount { get; }

    public int GetLength(Index dimension);

    public T GetElementAt(params int[] indecies);

    public bool TryGetElementAt([NotNullWhen(true)] out T? element, params int[] indecies);

    public void ForEach(Action<int[], T> action);
    public void LinearForEach(Action<int, T> action);
}

