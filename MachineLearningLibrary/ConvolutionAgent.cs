
namespace MachineLearningLibrary;

public sealed class ConvolutionAgent: IAgent
{

    private readonly ArrayImage<float> kernals;

    private readonly int[] inputLengths;
    private readonly int[] outputLengths;

    public ConvolutionAgent(ArrayImage<float> kernals, int[] inputLengths, int[] outputLengths)
    {
        this.kernals = kernals;
        this.inputLengths = inputLengths;
        this.outputLengths = outputLengths;
    }

    public int VariableCount() => kernals.ElementCount;

    public void AddAll(IReadOnlyList<float> values)
    {
        kernals.AssignByActualIndex((i, v) => v + values[i]);
    }

    public void Invoke(in IImage<float> image, out IImage<float> result)
    {
        for(int i = 0; i < inputLengths.Length; i++)
        {
            if(image.GetLength(i) != inputLengths[i])
            {
                throw new ArgumentException();
            }
        }

        IImage<float> tempImages = image;
        ArrayImage<float> resultValues = new(outputLengths);

        int[] offsets = new int[kernals.Rank];
        for(int i = 0; i < offsets.Length; i++)
        {
            offsets[i] = (inputLengths[i] - outputLengths[i] - kernals.GetLength(i)) / 2;
        }

        resultValues.AssignEach((resultIndecies, _) =>
        {
            int[] offsets2 = new int[kernals.Rank];
            for(int i = 0; i < offsets2.Length; i++)
            {
                offsets2[i] = offsets[i] + resultIndecies[i];
            }

            float average = 0;
            int countPlusOne = 1;

            kernals.ForEach((kernalIndecies, kernalElement) => 
            {
                int[] imageIndecies = new int[offsets2.Length];
                for(int i = 0; i < offsets2.Length; i++)
                {
                    imageIndecies[i] = offsets2[i] + kernalIndecies[i];
                }

                if(!tempImages.TryElementAt(out float imageElement, imageIndecies))
                {
                    return;
                }

                average += (kernalElement * imageElement - average) / countPlusOne;
                countPlusOne++;
            });

            return average;
        });

        result = resultValues;
    }
    
    //TODO: Add derivative part.
    public void Invoke(
        in IImage<float> image, 
        in IImage<float>? gradient, 
        out IImage<float> valueResult, 
        out IImage<float> derivativeResult, 
        int varIndex = -1)
    {
        //TODO: Assign default gradient.
        throw new NotImplementedException();
    }

    public void WriteToFile(BinaryWriter binWriter)
    {
        throw new NotImplementedException();
    }

    public static ConvolutionAgent ReadFromFile(BinaryReader binReader)
    {
        throw new NotImplementedException();
    }
}
