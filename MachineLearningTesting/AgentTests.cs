
using MachineLearningLibrary;

namespace MachineLearningTesting;

internal class AgentTests
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

        AffineLayer[] layers = new AffineLayer[weights.Length];
        for(int i = 0; i < weights.Length; i++)
        {
            layers[i] = new AffineLayer(weights[i]);
        }

        Agent agent = new(layers);

        string filename = "testSave.bin";

        agent.SaveToFile(path + filename);
        Agent loadedAgent = Agent.LoadFromFile(path + filename);

        Assert.That(agent.VariableCount(), Is.EqualTo(loadedAgent.VariableCount()));
    }
}
