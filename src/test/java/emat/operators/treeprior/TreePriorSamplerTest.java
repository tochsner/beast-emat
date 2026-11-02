package emat.operators.treeprior;

import beast.base.evolution.tree.TreeParser;
import org.junit.jupiter.api.Test;

import java.util.function.ToDoubleFunction;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TreePriorSamplerTest {

    private static final double MEAN = 1.5;
    private static final double SD = 0.4;

    private TreeParser createTree() {
        TreeParser tree = new TreeParser();
        tree.initByName("newick", "((a:1.0,b:1.0):1.0,(c:0.25,d:0.5):1.5)", "IsLabelledNewick", true, "adjustTipHeights", false);
        return tree;
    }

    @Test
    public void computesStatisticsOfTheTree() {
        double[] statistics = TreeStatistics.compute(this.createTree());

        assertEquals(Math.log(5.25), statistics[0], 1.0E-12);
        assertEquals(Math.log(2.0), statistics[1], 1.0E-12);
        assertEquals(Math.log(0.75), statistics[TreeStatistics.COUNT - 1], 1.0E-12);
    }

    @Test
    public void learnsTheTargetFromSweepsAndLeavesTheParametersAlone() {
        TreeParser tree = this.createTree();
        double[] current = {MEAN, 0.0};
        double[] lastSet = new double[2];
        ToDoubleFunction<double[]> evaluate = x -> {
            lastSet[0] = x[0];
            lastSet[1] = x[1];
            return -0.5 * Math.pow((x[0] - MEAN) / SD, 2) - 0.5 * Math.pow(x[1], 2);
        };

        TreePriorSampler sampler = new TreePriorSampler(2, 0, 1, 4, 200, 4000, 1.0);
        assertFalse(sampler.isFitted());
        for (int i = 0; i < 400; i++) {
            sampler.record(tree, new double[0], current, evaluate);
            assertEquals(current[0], lastSet[0], 0.0);
        }
        assertTrue(sampler.isFitted());

        double sum = 0.0;
        double sumSquares = 0.0;
        int n = 4000;
        for (int i = 0; i < n; i++) {
            double draw = sampler.sample(tree, new double[0])[0];
            sum += draw;
            sumSquares += draw * draw;
        }
        double mean = sum / n;
        assertEquals(MEAN, mean, 0.1);
        assertEquals(SD, Math.sqrt(sumSquares / n - mean * mean), 0.1);
    }

}
