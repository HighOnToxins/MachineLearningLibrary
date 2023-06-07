
namespace MachineLearningLibrary;

//TODO: IDEA: Create a one-axis-kernal convolution.
public sealed class Convolution2DAgent: IAgent
{

    private readonly float[][] kernal;

    private readonly int variableCount;

    private readonly int offsetX;
    private readonly int offsetY;

    private readonly int inputCount;

    //TODO: Consider adding a bias.

    public Convolution2DAgent(float[][] kernal, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
    {
        this.kernal = kernal;
        InputWidth = inputWidth;
        InputHeight = inputHeight;
        OutputWidth = outputWidth;
        OutputHeight = outputHeight;

        int kernal0length = kernal[0].Length;
        for(int x = 0; x < kernal.Length; x++)
        {
            if(kernal[x].Length != kernal0length)
            {
                throw new ArgumentException();
            }
        }

        variableCount = kernal.Length * kernal0length;
        inputCount = OutputWidth * OutputHeight;

        offsetX = (InputWidth - OutputWidth - KernalWidth) / 2;
        offsetY = (InputHeight - OutputHeight - KernalHeight) / 2;
    }

    public Convolution2DAgent(
        int kernalWidth, int kernalHeight, 
        int inputWidth, int inputHeight, 
        int outputWidth, int outputHeight,
        int rndRangeMin, int rndRangeMax)
    {
        InputWidth = inputWidth;
        InputHeight = inputHeight;
        OutputWidth = outputWidth;
        OutputHeight = outputHeight;

        Random random = new();

        kernal = new float[kernalWidth][];
        int kernal0length = kernal[0].Length;
        for(int x = 0; x < kernal.Length; x++)
        {
            kernal[x] = new float[kernalHeight];
            for(int y = 0; y < kernal0length; y++)
            {
                kernal[x][y] = random.Next(rndRangeMin, rndRangeMax);
            }
        }

        variableCount = kernal.Length * kernal0length;
        inputCount = OutputWidth * OutputHeight;

        offsetX = (InputWidth - OutputWidth - KernalWidth) / 2;
        offsetY = (InputHeight - OutputHeight - KernalHeight) / 2;
    }

    public int KernalWidth { get => kernal.Length; }
    public int KernalHeight { get => kernal[0].Length; }

    public int InputWidth { get; }
    public int InputHeight { get; }
    public int OutputWidth { get; }
    public int OutputHeight { get; }

    public int VariableCount() => variableCount;

    public void AddAll(IReadOnlyList<float> values)
    {
        for(int x = 0; x < InputWidth; x++)
        {
            for(int y = 0; y < InputHeight; y++)
            {
                kernal[x][y] += values[x * InputHeight + y];
            }
        }
    }

    public void Invoke(in IReadOnlyList<float> value, out IReadOnlyList<float> result)
    {
        if(value.Count != OutputHeight * OutputWidth) throw new NotImplementedException();

        float[] valueResult = new float[OutputWidth * OutputHeight];

        for(int outX = 0; outX < OutputWidth; outX++)
        {
            for(int outY = 0; outY < OutputHeight; outY++)
            {
                float average = 0;
                int count = 0;

                int outOffsetX = offsetX + outX;
                int outoffsetY = offsetY + outY;

                for(int kerX = 0; kerX < KernalWidth; kerX++)
                {
                    for(int kerY = 0; kerY < KernalHeight; kerY++)
                    {
                        int valX = outOffsetX + kerX;
                        int valY = outoffsetY + kerY;

                        if(valX < 0 || InputWidth < valX || valY < 0 || InputHeight < valY)
                        {
                            continue;
                        }

                        int valI = valX * InputHeight + valY;
                        average += kernal[kerX][kerY] * value[valI];
                        count++;
                    }
                }

                valueResult[outX * OutputHeight + outY] = Math.Max(0, average / count);
            }
        }

        result = valueResult;
    }

    public void Invoke(
        in IReadOnlyList<float> value,
        in IReadOnlyList<float>? gradient, 
        out IReadOnlyList<float> valueOutput, 
        out IReadOnlyList<float> derivativeResult, 
        int varIndex = -1)
    {
        Invoke(value, out IReadOnlyList<float> valueResult);

        float[] derivative = new float[OutputWidth * OutputHeight];

        if(gradient is not null)
        {
            if(gradient.Count != InputWidth * InputHeight) throw new ArgumentException();

            for(int outX = 0; outX < OutputWidth; outX++)
            {
                for(int outY = 0; outY < OutputHeight; outY++)
                {
                    int outI = outX * OutputHeight + outY;
                    if(valueResult[outI] == 0) continue;

                    float average = 0;
                    int count = 0;

                    int outOffsetX = offsetX + outX;
                    int outoffsetY = offsetY + outY; 

                    for(int kerX = 0; kerX < KernalWidth; kerX++)
                    {
                        for(int kerY = 0; kerY < KernalHeight; kerY++)
                        {
                            int valX = outOffsetX + kerX;
                            int valY = outoffsetY + kerY;
                            int valI = valX * InputHeight + valY;

                            if(0 <= valI && valI < inputCount)
                            {
                                average += kernal[kerX][kerY] * gradient[valI];
                                count++;
                            }
                        }
                    }

                    derivative[outI] = average / count;

                    int kerDifX = varIndex / KernalWidth;
                    int kerDifY = varIndex % KernalWidth;
                    int valDifX = outOffsetX + kerDifX;
                    int valDifY = outoffsetY + kerDifY;
                    int valDifI = valDifX * InputHeight + valDifY;

                    if(0 <= valDifI && valDifI < inputCount)
                    {
                        derivative[outI] += value[valDifI];
                    }
                }
            }
        }

        valueOutput = valueResult;
        derivativeResult = derivative;
    }

    public void WriteToFile(BinaryWriter binWriter)
    {
        binWriter.Write(KernalWidth); 
        binWriter.Write(KernalHeight);

        binWriter.Write(InputWidth);
        binWriter.Write(InputHeight);

        binWriter.Write(OutputWidth);
        binWriter.Write(OutputHeight);

        for(int x = 0; x < KernalWidth; x++)
        {
            for(int y = 0; y < KernalHeight; y++)
            {
                binWriter.Write(kernal[x][y]);
            }
        }
    }

    public static IAgent ReadFromFile(BinaryReader binReader)
    {
        int kernalWidth = binReader.ReadInt32();
        int kernalHeight = binReader.ReadInt32();

        int inputWidth = binReader.ReadInt32();
        int inputHeight = binReader.ReadInt32();

        int outputWidth = binReader.ReadInt32();
        int outputHeight = binReader.ReadInt32();

        float[][] kernal = new float[kernalWidth][];
        for(int x = 0; x < kernalWidth; x++)
        {
            kernal[x] = new float[kernalHeight];
            for(int y = 0; y < kernalHeight; y++)
            {
                kernal[x][y] = binReader.ReadSingle();
            }
        }

        return new Convolution2DAgent(kernal, inputWidth, inputHeight, outputWidth, outputHeight);
    }
}
