
using System;
using System.Reflection;

namespace MachineLearningLibrary.Layers;

public sealed class AffineLayer: ILayer
{
    private readonly float[][] matrix; //out * in
    private readonly float[] bias; //out

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
        for(int outI = 0; outI < OutputSize; outI++)
        {
            matrix[outI] = new float[weights.GetLength(1) - 1];
            for(int inI = 0; inI < InputSize; inI++)
            {
                matrix[outI][inI] = weights[outI, inI];
            }
        }

        bias = new float[OutputSize];
        for(int i = 0; i < OutputSize; i++)
        {
            bias[i] = weights[i, InputSize];
        }
    }

    public int VariableLength { 
        get => InputSize * OutputSize + bias.Length;
    }

    public int InputSize { get => matrix[0].Length; }

    public int OutputSize { get => matrix.Length; }

    public void AddGradient(float[] gradient)
    {
        for(int outI = 0; outI < OutputSize; outI++)
        {
            for(int inI = 0; inI < InputSize; inI++)
            {
                matrix[outI][inI] += gradient[outI*OutputSize + inI];
            }
        }

        for(int outI = 0; outI < OutputSize; outI++)
        {
            bias[outI] += gradient[OutputSize * InputSize - 1 + outI];
        }
    }

    public IReadOnlyList<float> ForwardPass(IReadOnlyList<float> data)
    {
        if(data.Count != InputSize)
        {
            throw new ArgumentException();
        }

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

        // - - - CAN BE REPLACED BY THIS, but is very slow unless the ComputeGradient function is optimized - - - 
        //IReadOnlyList<float> emptyGradient = Array.Empty<float>();
        //ComputeGradient(-1, ref emptyGradient, ref data);
        //return data;
    }

    //TODO: find a way to optimize compute gradient from gradient, such as to not compute values where gradient are zero.
    public void ComputeGradient(int index, ref IReadOnlyList<float> gradient, ref IReadOnlyList<float> data)
    {
        if(gradient.Count != InputSize && data.Count != InputSize)
        {
            throw new ArgumentException();
        }

        float[] gradientResult = new float[OutputSize];
        float[] dataResult = new float[OutputSize];

        bool matrixFlag = 0 <= index && index < InputSize * OutputSize;
        int outputIndex = index / OutputSize;
        int inputIndex = index % OutputSize;

        bool biasFlag = InputSize * OutputSize < index && index < (InputSize + 1) * OutputSize;
        int bOutputIndex = index - InputSize * OutputSize;

        for(int outI = 0; outI < OutputSize; outI++)
        {
            for(int inI = 0; inI < InputSize; inI++)
            {
                dataResult[outI] += matrix[outI][inI] * data[inI];
                gradientResult[outI] += matrix[outI][inI] * gradient[inI];
            }

            dataResult[outI] += bias[outI];

            float differentiatedMatrixOIPart = matrixFlag && outI == outputIndex ? data[inputIndex] : 0;
            int differentiatedBias = biasFlag && outI == bOutputIndex ? 1 : 0;
            gradientResult[outI] += differentiatedMatrixOIPart + differentiatedBias;

            gradientResult[outI] = dataResult[outI] < 0 ? 0 : gradientResult[outI]; 

            dataResult[outI] = Math.Max(0, dataResult[outI]);

            //if dataResult[outI] == 0, then we might have a problem, quick check.
            if(dataResult[outI] == 0) throw new ArgumentException($"dataResult[{outI}] = 0");
        }

        gradient = gradientResult;
        data = dataResult; 
    }
}
