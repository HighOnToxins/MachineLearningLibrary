
namespace MachineLearningLibrary;

public sealed class Convolution2DAgent: IAgent
{

    private readonly float[][] kernal;

    private readonly int variableCount;

    private readonly int offsetX;
    private readonly int offsetY;

    public Convolution2DAgent(float[][] kernal, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
    {
        this.kernal = kernal;
        InputWidth = inputWidth;
        InputHeight = inputHeight;
        OutputWidth = outputWidth;
        OutputHeight = outputHeight;

        int kernal0length = kernal[0].Length;
        for(int i = 0; i < kernal.Length; i++)
        {
            if(kernal[i].Length != kernal0length)
            {
                throw new ArgumentException();
            }
        }

        variableCount = kernal.Length * kernal0length;

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
        float[] valueResult = new float[OutputWidth * OutputHeight];

        for(int outX = 0; outX < OutputWidth; outX++)
        {
            for(int outY = 0; outY < OutputHeight; outY++)
            {
                float average = 0;
                int countPlusOne = 1;

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
                        float v = kernal[kerX][kerY] * value[valI];
                        average += (v - average) / countPlusOne;
                    }
                }

                valueResult[outX * OutputHeight + outY] = average;
            }
        }

        result = valueResult;
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
