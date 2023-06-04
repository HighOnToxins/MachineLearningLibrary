
using MachineLearningLibrary;

namespace MachineLearningTesting;

internal class FileTests
{

    public const string path = "../../../savedAgents/";

    [SetUp]
    public void CreateSavedAgentsFolder()
    {
        Directory.CreateDirectory(path);
    }

    [Test]
    public void AgentIsSavedAndReadProperly()
    {

        float[][,] weights = new float[][,]{
            new float[,]{
                {1f, 2f, 1f},
                {3f, 4f, 1f},
            },
            new float[,]{
                {5f, 6f, 1f},
                {7f, 8f, 1f},
            },
            new float[,]{
                {9f, 10f, 1f},
                {11f, 12f, 1f},
            },
        };

        AffineAgent[] layers = new AffineAgent[weights.Length];
        for(int i = 0; i < weights.Length; i++)
        {
            layers[i] = new AffineAgent(weights[i]);
        }

        AgentComposite agent = new(layers);

        string filename = "testSave.bin";

        agent.SaveToFile(path + filename);
        AgentComposite loadedAgent = AgentComposite.LoadFromFile(path + filename);

        Assert.That(agent.VariableCount(), Is.EqualTo(loadedAgent.VariableCount()));

        for(int i = -10; i < 10; i++)
        {
            for(int j = -10; j < 10; j++)
            {

                IReadOnlyList<float> data = new float[] { 
                    i, j
                };

                agent.Invoke(data, out IReadOnlyList<float> result);
                loadedAgent.Invoke(data, out IReadOnlyList<float> result2);

                Assert.That(result, Is.EquivalentTo(result2));

            }
        }

    }
}
