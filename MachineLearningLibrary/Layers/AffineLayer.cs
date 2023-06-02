
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

    internal override void AddValueAt(int index, float value)
    {
        if(0 < index && index < InputSize * OutputSize)
        {
            int outputIndex = index / OutputSize;
            int inputIndex = index % OutputSize;
            matrix[outputIndex][inputIndex] = value;
        }
        else if(index < (InputSize+1) * OutputSize)
        {
            int outputIndex = index - InputSize * OutputSize;
            bias[outputIndex] = value;
        }
        else
        {
            throw new IndexOutOfRangeException();
        }
    }

    internal override IReadOnlyList<float> ForwardPass(IReadOnlyList<float> data)
    {
        float[] result = new float[OutputSize];

        for(int outI = 0; outI < OutputSize; outI++)
        {
            for(int inI = 0; inI < InputSize; inI++)
            {
                result[outI] += matrix[outI][inI] * data[inI];
            }
            result[outI] += bias[outI];
            result[outI] = Math.Max(0, result[outI]);
        }

        return result;
    }

    //TODO: find a way to optimize compute gradient from gradient, such as to not compute values where gradient are zero.
    //TODO: check if the out gradient overrides the gradient while the function is on-going if they are the same array.
    internal override void ComputeGradient(int index, ref IReadOnlyList<float> gradient, ref IReadOnlyList<float> data)
    {
        float[] gradientResult = new float[OutputSize];
        float[] dataResult = new float[OutputSize];

        bool matrixFlag = 0 < index && index < InputSize * OutputSize;
        int outputIndex = index / OutputSize;
        int inputIndex = index % OutputSize;

        bool biasFlag = InputSize * OutputSize < index && index < (InputSize + 1) * OutputSize;
        int bOutputIndex = index - InputSize * OutputSize;

        for(int outI = 0; outI < OutputSize; outI++)
        {
            for(int inI = 0; inI < InputSize; inI++)
            {
                dataResult[outI] += matrix[outI][inI] * data[inI];

                int differentiatedMatrixOI = matrixFlag && inI == inputIndex && outI == outputIndex ? 1 : 0;
                gradientResult[outI] += matrix[outI][inI] * gradient[inI] + differentiatedMatrixOI * data[inI];
            }

            dataResult[outI] += bias[outI];

            gradientResult[outI] = dataResult[outI] <= 0 ? 0 : gradientResult[outI]; 
            
            //if dataResult[outI] == 0, then we might have a problem, quick check.
            if(dataResult[outI] == 0) throw new ArgumentException($"dataResult[{outI}] = 0");

            int differentiatedBias = biasFlag && outI == bOutputIndex ? 1 : 0;
            gradientResult[outI] += differentiatedBias;

            dataResult[outI] = Math.Max(0, dataResult[outI]);
        }

        gradient = gradientResult;
        data = dataResult; 
    }
}
