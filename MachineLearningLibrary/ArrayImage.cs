
using System.Diagnostics.CodeAnalysis;

namespace MachineLearningLibrary;

public class ArrayImage<T>: IImage<T> where T : notnull
{
    private readonly int[] lengths;
    private readonly int[] stepSizes;

    private readonly T[] array;

    public ArrayImage(params int[] lengths)
    {
        this.lengths = lengths;
        stepSizes = ComputeStepSizes(lengths);
        array = new T[stepSizes[^1]];
    }

    private static int[] ComputeStepSizes(int[] lengths)
    {
        if(lengths.Length == 0)
        {
            return Array.Empty<int>();
        }

        int[] stepSizes = new int[lengths.Length];
        stepSizes[0] = lengths[0];

        for(int i = 1; i < lengths.Length; i++)
        {
            stepSizes[i] = stepSizes[i - 1] * lengths[i];
        }

        return stepSizes;
    }

    public int Rank => lengths.Length;

    public int ElementCount => stepSizes[^1];

    public int GetLength(int dimension) => lengths[dimension];

    public T GetElementAt(params int[] indecies) => array[GetActualIndex(indecies)];

    public bool TryGetElementAt(int[] indecies, [NotNullWhen(true)] out T? element)
    {
        try
        {
            element = GetElementAt(indecies);
            return true;
        }
        catch
        {
            element = default;
            return false;
        }
    }

    public void AssignElementAt(T value, params int[] indecies) => array[GetActualIndex(indecies)] = value;

    public void ForEach(Action<int[], T> action)
    {
        for(int i = 0; i < array.Length; i++)
        {
            action.Invoke(GetIndecies(i), array[i]);
        }
    }

    public void AssignEeach(Func<int[], T, T> assignment)
    {
        for(int i = 0; i < array.Length; i++)
        {
            array[i] = assignment.Invoke(GetIndecies(i), array[i]);
        }
    }

    public void AssignByActualIndex(Func<int, T, T> assignment)
    {
        for(int i = 0; i < array.Length; i++)
        {
            array[i] = assignment.Invoke(i, array[i]);
        }
    }

    private int GetActualIndex(int[] indecies)
    {
        int index = indecies[0];
        int maxDimension = Math.Min(Rank, indecies.Length);
        for(int i = 1; i < maxDimension; i++)
        {
            index += indecies[i] * stepSizes[i - 1];
        }

        return index;
    }

    private int[] GetIndecies(int actualIndex)
    {
        int[] indecies = new int[Rank];
        indecies[0] = actualIndex % lengths[^1];

        for(int i = 1; i < lengths.Length; i++)
        {
            indecies[i] = actualIndex / stepSizes[i - 1] % lengths[i];
        }

        return indecies;
    }

}