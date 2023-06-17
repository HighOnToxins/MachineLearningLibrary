
using System.Runtime.CompilerServices;

namespace MachineLearningLibrary;

public sealed class AffineAgent : IAgent
{
    private readonly float[][] matrix; //out * in
    private readonly float[] bias; //out

    private readonly int matrixCount;

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

        matrixCount = matrix.Length * matrix0length;
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

        matrixCount = matrix.Length * matrix[0].Length;
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
        => matrixCount + bias.Length;

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
        Invoke(value, out IReadOnlyList<float> valueResult);
        float[] gradientResult = new float[OutputSize];

        if(gradient is not null)
        {
            if(gradient.Count != InputSize) throw new ArgumentException();
            
            for(int outI = 0; outI < OutputSize; outI++)
            {
                if(valueResult[outI] == 0) continue;

                for(int inI = 0; inI < InputSize; inI++)
                {
                    gradientResult[outI] += matrix[outI][inI] * gradient[inI];
                }
            }
        }

        //Speed test v1: 
        //int matrixFlag = (0 <= varIndex).ToByte() * (varIndex < matrixCount).ToByte();
        //int mOutputIndex = varIndex / OutputSize;
        //int inputIndex = varIndex % OutputSize;

        //int biasFlag = (matrixCount < varIndex).ToByte() * (varIndex < VariableCount()).ToByte();
        //int bOutputIndex = varIndex - matrixCount;

        //int outputIndex = matrixFlag * mOutputIndex + biasFlag * bOutputIndex;
        //gradientResult[outputIndex] += matrixFlag * value[inputIndex] + biasFlag;

        //Speed test v2
        if(0 <= varIndex && varIndex < matrixCount)
        {
            int outputIndex = varIndex / OutputSize;
            int inputIndex = varIndex % OutputSize;
            gradientResult[outputIndex] += value[inputIndex];
        }
        else if(matrixCount <= varIndex && varIndex < VariableCount())
        {
            int outputIndex = varIndex - matrixCount;
            gradientResult[outputIndex]++;
        }

        valueOut = valueResult;
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
