
namespace MachineLearningLibrary;

public class MultiArray<T>: IArray<T>
{
    private readonly int[] lengths;
    private readonly int[] stepSizes;

    private readonly T[] array;

    public MultiArray(params int[] lengths)
    {
        this.lengths = lengths;
        stepSizes = ComputeStepSizes(lengths);
        array = new T[stepSizes[^1] * lengths[^1]];
    }

    private static int[] ComputeStepSizes(int[] lengths)
    {
        if(lengths.Length == 0)
        {
            return Array.Empty<int>();
        }

        int[] stepSizes = new int[lengths.Length - 1];
        stepSizes[0] = lengths[0];

        for(int i = 1; i < lengths.Length - 1; i++)
        {
            stepSizes[i] = stepSizes[i - 1] * lengths[i];
        }
        return stepSizes;
    }

    public int Rank => lengths.Length;

    public int GetLength(int dimension) => lengths[dimension];
    public T GetElementAt(params int[] indecies) => array[GetActualIndex(indecies)];
    public void AssignElementAt(T value, params int[] indecies) => array[GetActualIndex(indecies)] = value;

    public void AssignEeach(Func<int[], T> assignment)
    {
        for(int i = 0; i < array.Length; i++)
        {
            array[i] = assignment.Invoke(GetIndecies(i));
        }
    }

    private int GetActualIndex(int[] indecies)
    {
        if(lengths.Length != indecies.Length || lengths.Length == 0)
        {
            throw new IndexOutOfRangeException();
        }

        int index = indecies[0];
        for(int i = 1; i < indecies.Length; i++)
        {
            index += indecies[i] * stepSizes[i - 1];
        }

        return index;
    }

    private int[] GetIndecies(int actualIndex)
    {
        int[] indecies = new int[Rank];
        indecies[^1] = actualIndex % lengths[^1];

        for(int i = 0; i < lengths.Length; i++)
        {
            actualIndex = actualIndex / lengths[i];
            indecies[i] = actualIndex % lengths[i];
        }

        return indecies;
    }
}