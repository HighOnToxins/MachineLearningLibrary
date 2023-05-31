
namespace MachineLearningLibrary.Layers;

public sealed class AffineLayer: Layer
{
    private readonly float[][] matrix; //out * in
    private readonly float[] bias;

    //TODO: add the ability to add your own activation function / activation derivative.

    public AffineLayer(float[][] matrix, float[] bias)
    {
        this.matrix = matrix; 
        this.bias = bias;

        int matrix0length = matrix[0].Length;

        for(int i = 0; i < matrix.Length; i++)
        {
            if(matrix[i].Length != matrix0length)
            {
                throw new ArgumentException();
            }
        }

        if(bias.Length != matrix0length)
        {
            throw new ArgumentException();
        }
    }

    public AffineLayer(float[,] weights)
    {
        matrix = new float[weights.GetLength(0)][];
        for(int i = 0; i < OutputSize; i++)
        {
            matrix[i] = new float[weights.GetLength(1) - 1];
            for(int j = 0; j < InputSize; j++)
            {
                matrix[i][j] = weights[i, j];
            }
        }

        bias = new float[weights.GetLength(1)];
        for(int i = 0; i < OutputSize; i++)
        {
            bias[i] = weights[i, weights.GetLength(0) - 1];
        }
    }

    public override int VariableLength { 
        get => InputSize * OutputSize + bias.Length;
    }

    public override int InputSize { get => matrix[0].Length; }

    public override int OutputSize { get => matrix.Length; }

    internal override IReadOnlyList<float> ForwardPass(IReadOnlyList<float> data)
    {
        float[] result = new float[OutputSize];

        for(int y = 0; y < OutputSize; y++)
        {
            for(int x = 0; x < InputSize; x++)
            {
                result[y] += matrix[y][x] * data[x];
            }
            result[y] += bias[y];
            result[y] = Math.Max(0, result[y]);
        }

        return result;
    }

    internal override IReadOnlyList<float> ComputeGradient(int index, IReadOnlyList<float> data)
    {
        float[] result = new float[OutputSize];

        if(index < InputSize * OutputSize)
        {
            int outputIndex = index / OutputSize;
            int inputIndex = index % OutputSize;
            result[outputIndex] = data[inputIndex];
        }
        else if(index < InputSize * OutputSize + OutputSize)
        {
            int outputIndex = index - InputSize * OutputSize;
            result[outputIndex] = 1;
        }

        return result;
    }

    internal override IReadOnlyList<float> ComputeGradientFromGradient(int index, IReadOnlyList<float> gradient, IReadOnlyList<float> data)
    {
        float[] result = new float[OutputSize];

        bool matrixFlag = index < InputSize * OutputSize;
        int matrixY = index / OutputSize;
        int matrixX = index % OutputSize;
        int biasIndex = index - InputSize * OutputSize;

        for(int outI = 0; outI < OutputSize; outI++)
        {
            for(int inI = 0; inI < InputSize; inI++)
            {
                // (f(x)*g(x))' = f(x)*g'(x) + f'(x)*g(x)
                int differentiatedMatrixYX = matrixFlag && inI == matrixX && outI == matrixY ? 1 : 0;
                result[outI] += matrix[outI][inI] * gradient[inI] + differentiatedMatrixYX * data[inI]; 
            }

            int differentiatedBias = !matrixFlag && outI == biasIndex ? 1 : 0;
            result[outI] = Math.Max(0, result[outI]) + differentiatedBias;
        }

        return result;
    }

    internal override void AddValueAt(int index, float value)
    {
        if(index < InputSize * OutputSize)
        {
            int outputIndex = index / OutputSize;
            int inputIndex = index % OutputSize;
            matrix[outputIndex][inputIndex] = value;
        }
        else if(index < InputSize * OutputSize + OutputSize)
        {
            int outputIndex = index - InputSize * OutputSize;
            bias[outputIndex] = value;
        }
        else
        {
            throw new IndexOutOfRangeException();
        }
    }
}
