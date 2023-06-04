
namespace MachineLearningLibrary;

public sealed class AffineAgent : IAgent
{
    private readonly float[][] matrix; //out * in
    private readonly float[] bias; //out

    //TODO: add the ability to add your own activation function / activation derivative.

    public AffineAgent(float[][] matrix, float[] bias)
    {
        this.matrix = matrix;
        this.bias = bias;

        int matrix0length = matrix[0].Length;

        for (int i = 0; i < matrix.Length; i++)
        {
            if (matrix[i].Length != matrix0length)
            {
                throw new ArgumentException();
            }
        }

        if (bias.Length != matrix0length)
        {
            throw new ArgumentException();
        }
    }

    public AffineAgent(float[,] weights)
    {
        matrix = new float[weights.GetLength(0)][];
        for (int outI = 0; outI < OutputSize; outI++)
        {
            matrix[outI] = new float[weights.GetLength(1) - 1];
            for (int inI = 0; inI < InputSize; inI++)
            {
                matrix[outI][inI] = weights[outI, inI];
            }
        }

        bias = new float[OutputSize];
        for (int i = 0; i < OutputSize; i++)
        {
            bias[i] = weights[i, InputSize];
        }
    }

    public AffineAgent(float[] variables)
    {
        throw new NotImplementedException();
    }

    public AffineAgent(int inputCount, int outputCount)
    {
        throw new NotImplementedException();
    }

    public AffineAgent(int inputCount, int outputCount, int rangeMin, int rangeMax)
    {
        throw new NotImplementedException();
    }

    public int VariableCount()
        => InputSize * OutputSize + bias.Length;

    public int InputSize { get => matrix[0].Length; }

    public int OutputSize { get => matrix.Length; }

    public void AddAll(IReadOnlyList<float> values)
    {
        for (int outI = 0; outI < OutputSize; outI++)
        {
            for (int inI = 0; inI < InputSize; inI++)
            {
                matrix[outI][inI] += values[outI * OutputSize + inI];
            }
        }

        for (int outI = 0; outI < OutputSize; outI++)
        {
            bias[outI] += values[OutputSize * InputSize - 1 + outI];
        }
    }

    public void Invoke(
        in IReadOnlyList<float> value,
        in IReadOnlyList<float>? gradient,
        out IReadOnlyList<float> valueOut,
        out IReadOnlyList<float> gradientOut,
        int varIndex = -1)
    {
        IReadOnlyList<float> gradientOrDefault = gradient ?? new float[InputSize];

        if (value.Count != InputSize || gradientOrDefault.Count != InputSize)
        {
            throw new ArgumentException();
        }

        float[] gradientResult = new float[OutputSize];
        float[] dataResult = new float[OutputSize];

        bool matrixFlag = 0 <= varIndex && varIndex < InputSize * OutputSize;
        int outputIndex = varIndex / OutputSize;
        int inputIndex = varIndex % OutputSize;

        bool biasFlag = InputSize * OutputSize < varIndex && varIndex < (InputSize + 1) * OutputSize;
        int bOutputIndex = varIndex - InputSize * OutputSize;

        for (int outI = 0; outI < OutputSize; outI++)
        {
            for (int inI = 0; inI < InputSize; inI++)
            {
                dataResult[outI] += matrix[outI][inI] * value[inI];
                gradientResult[outI] += matrix[outI][inI] * gradientOrDefault[inI];
            }

            dataResult[outI] += bias[outI];

            float differentiatedMatrixOIPart = matrixFlag && outI == outputIndex ? value[inputIndex] : 0;
            int differentiatedBias = biasFlag && outI == bOutputIndex ? 1 : 0;
            gradientResult[outI] += differentiatedMatrixOIPart + differentiatedBias;

            dataResult[outI] = Math.Max(0, dataResult[outI]);
            gradientResult[outI] = dataResult[outI] < 0 ? 0 : gradientResult[outI];

            //if dataResult[outI] == 0, then we might have a problem, quick check.
            if (dataResult[outI] == 0) throw new ArgumentException($"dataResult[{outI}] = 0"); //TODO: Remove dataResult == 0 test.
        }

        valueOut = dataResult;
        gradientOut = gradientResult;
    }


    public void Invoke(in IReadOnlyList<float> value, out IReadOnlyList<float> valueResult)
    {
        if (value.Count != InputSize)
        {
            throw new ArgumentException();
        }

        float[] result = new float[OutputSize];

        for (int outI = 0; outI < OutputSize; outI++)
        {
            //TODO: For larger AI, compute matricies threaded.
            for (int inI = 0; inI < InputSize; inI++)
            {
                result[outI] += matrix[outI][inI] * value[inI];
            }
            result[outI] += bias[outI];
            result[outI] = Math.Max(0, result[outI]);
        }

        valueResult = result;
    }

    public void WriteToFile(BinaryWriter binWriter)
    {
        binWriter.Write(OutputSize);
        binWriter.Write(InputSize);

        for(int outI = 0; outI < OutputSize; outI++)
        {
            for(int inI = 0; inI < InputSize; inI++)
            {
                binWriter.Write(matrix[outI][inI]);
            }
        }

        for(int outI = 0; outI < OutputSize; outI++)
        {
            binWriter.Write(bias[outI]);
        }
    }

    internal static IAgent ReadFromFile(BinaryReader binReader)
    {
        int outputSize = binReader.ReadInt32();
        int inputSize = binReader.ReadInt32();

        float[][] matrix = new float[outputSize][];

        for(int outI = 0; outI < outputSize; outI++)
        {
            matrix[outI] = new float[inputSize];
            for(int inI = 0; inI < inputSize; inI++)
            {
                matrix[outI][inI] = binReader.ReadSingle();
            }
        }

        float[] bias = new float[outputSize];

        for(int outI = 0; outI < outputSize; outI++)
        {
            bias[outI] = (float)binReader.ReadSingle();
        }

        return new AffineAgent(matrix, bias);
    }
}
