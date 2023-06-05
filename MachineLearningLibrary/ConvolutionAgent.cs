
using System;

namespace MachineLearningLibrary;

public sealed class ConvolutionAgent
{

    private readonly ArrayImage<float>[] kernals;

    private readonly int[] inputLengths;
    private readonly int[] outputLengths;

    public ConvolutionAgent(ArrayImage<float>[] kernals, int[] inputLengths, int[] outputLengths)
    {
        this.kernals = kernals;
        this.inputLengths = inputLengths;
        this.outputLengths = outputLengths;
    }

    public int VariableCount()
    {
        throw new NotImplementedException();
    }

    public void AddAll(IReadOnlyList<float> values)
    {
        for(int i = 0; i < kernals.Length; i++)
        {
            int currentStepSize = 0;
            kernals[i].AssignByActualIndex((varI, val) =>
            {
                return val + values[i*currentStepSize + varI];
            });
            currentStepSize += kernals[i].ElementCount;
        }
    }

    //TODO: Create an IMultiArray for determining that the input has the correct size and such.
    public void Invoke(in IImage<float>[] images, out IImage<float>[] result)
    {
        IImage<float>[] tempImages = images;
        ArrayImage<float>[] resultValues = new ArrayImage<float>[tempImages.Length*kernals.Length];

        int[] offsets = new int[kernals[0].Rank];
        for(int i = 0; i < offsets.Length; i++)
        {
            offsets[i] = (inputLengths[i] - outputLengths[i] - kernals[0].GetLength(i)) / 2;
        }

        for(int imageNum = 0; imageNum < tempImages.Length; imageNum++)
        {
            for(int kernalNum = 0; kernalNum < kernals.Length; kernalNum++)
            {
                int resultIndex = imageNum * kernals.Length + kernalNum;
                resultValues[resultIndex] = new ArrayImage<float>(outputLengths);
                resultValues[resultIndex].AssignEeach((resultIndecies, _) =>
                {
                    float average = 0;
                    int countPlusOne = 1;

                    kernals[kernalNum].ForEach((kernalIndecies, kernalElement) => {

                        int[] imageIndecies = new int[kernalIndecies.Length];
                        for(int i = 0; i < offsets.Length; i++)
                        {
                            imageIndecies[i] = offsets[i] + resultIndecies[i] + kernalIndecies[i];
                        }

                        if(!tempImages[imageNum].TryGetElementAt(imageIndecies, out float imageElement))
                        {
                            imageElement = 0;
                        }

                        average += (kernalElement * imageElement - average) / countPlusOne;
                        countPlusOne++;
                    });

                    return average;
                });
            }
        }
        
        result = resultValues;
    }

    public void Invoke(in IReadOnlyList<float> value, in IReadOnlyList<float>? gradient, out IReadOnlyList<float> valueResult, out IReadOnlyList<float> derivativeResult, int varIndex = -1)
    {
        throw new NotImplementedException();
    }

    public void WriteToFile(BinaryWriter binWriter)
    {
        throw new NotImplementedException();
    }
}
