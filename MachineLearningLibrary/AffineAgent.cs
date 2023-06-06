
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
        in IImage<float> value,
        in IImage<float>? gradient,
        out IImage<float> valueOut,
        out IImage<float> gradientOut,
        int varIndex = -1)
    {
        IImage<float> gradientOrDefault = gradient ?? new ArrayImage<float>(InputSize);

        if (value.GetLength(0) != InputSize || gradientOrDefault.GetLength(0) != InputSize)
        {
            throw new ArgumentException();
        }

        ArrayImage<float> gradientResult = new(OutputSize);
        ArrayImage<float> dataResult = new(OutputSize);

        bool matrixFlag = 0 <= varIndex && varIndex < InputSize * OutputSize;
        int outputIndex = varIndex / OutputSize;
        int inputIndex = varIndex % OutputSize;

        bool biasFlag = InputSize * OutputSize < varIndex && varIndex < (InputSize + 1) * OutputSize;
        int bOutputIndex = varIndex - InputSize * OutputSize;

        for (int outI = 0; outI < OutputSize; outI++)
        {
            value.LinearForEach((inI, v) =>
            {
                dataResult.AssignElementAt(dataResult.GetElementAt(outI) + matrix[outI][inI] * v, outI);
            });
            dataResult.AssignElementAt(dataResult.GetElementAt(outI) + bias[outI], outI);

            if(dataResult.GetElementAt(outI) <= 0)
            {
                dataResult.AssignElementAt(0, outI);
                continue;
            }

            gradientOrDefault.LinearForEach((inI, g) => 
            {
                gradientResult.AssignElementAt(gradientResult.GetElementAt(outI) + matrix[outI][inI] * g, outI);
            });

            float differentiatedMatrixOIPart = matrixFlag && outI == outputIndex ? value.GetElementAt(inputIndex) : 0;
            int differentiatedBias = biasFlag && outI == bOutputIndex ? 1 : 0;
            gradientResult.AssignElementAt(gradientResult.GetElementAt(outI) + differentiatedMatrixOIPart + differentiatedBias, outI);
        }

        valueOut = dataResult;
        gradientOut = gradientResult;
    }


    public void Invoke(in IImage<float> value, out IImage<float> valueResult)
    {
        if (value.ElementCount != InputSize)
        {
            throw new ArgumentException();
        }

        ArrayImage<float> result = new(OutputSize);

        for (int outI = 0; outI < OutputSize; outI++)
        {
            //TODO: For larger AI, compute matricies threaded.
            value.LinearForEach((inI, v) =>
            {
                result.AssignElementAt(result.GetElementAt(outI) + matrix[outI][inI] * v, outI);
            });

            result.AssignElementAt(Math.Max(0, result.GetElementAt(outI) + bias[outI]), outI);
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
