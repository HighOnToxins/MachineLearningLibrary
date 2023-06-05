
namespace MachineLearningLibrary;

public interface IArray<T>
{
    public int Rank { get; }

    public int GetLength(int dimension);

    public T GetElementAt(params int[] indecies);

}

