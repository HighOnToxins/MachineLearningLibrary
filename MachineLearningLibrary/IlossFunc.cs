
namespace MachineLearningLibrary;

public interface ILossFunc : IDifferentiable<IReadOnlyList<float>, float>
{

    public int Answer { get; set; }

}
