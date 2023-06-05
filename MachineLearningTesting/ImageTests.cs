using MachineLearningLibrary;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningTesting;

internal sealed class ImageTests
{
    [Test]
    public void ImagesWorkProperly()
    {
        int[,,] ints = new int[3, 4, 5]; 
        for(int i = 0; i < ints.GetLength(0); i++)
        {
            for(int j = 0; j < ints.GetLength(1); j++)
            {
                for(int k = 0; k < ints.GetLength(1); k++)
                {
                    ints[i, j, k] = i * j + j * k - i * k;
                }
            }
        }

        ArrayImage<int> image = new(3, 4, 5);
        for(int i = 0; i < ints.GetLength(0); i++)
        {
            for(int j = 0; j < ints.GetLength(1); j++)
            {
                for(int k = 0; k < ints.GetLength(1); k++)
                {
                    image.AssignElementAt(ints[i, j, k], i, j, k);
                    Assert.That(image.GetElementAt(i, j, k), Is.EqualTo(ints[i, j, k]));
                }
            }
        }

    }

}
