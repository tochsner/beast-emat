package emat;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeIntervals;
import beast.base.evolution.tree.TreeParser;
import beast.base.util.Randomizer;
import emat.tree.InsertionSortTreeIntervals;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class InsertionSortTreeIntervalsTest {

    private static final int NUM_TIPS = 50;
    private static final int NUM_MOVES = 2000;

    /**
     * Repeatedly moves the height of a random internal node of a serially sampled tree, with
     * some tips sampled at the same time, and checks that the intervals and lineage counts
     * match those of TreeIntervals after every move.
     */
    @Test
    public void testMatchesTreeIntervals() {
        this.assertMatchesTreeIntervals(new InsertionSortTreeIntervals());
    }

    private void assertMatchesTreeIntervals(TreeIntervals actual) {
        Randomizer.setSeed(1);

        TreeParser tree = new TreeParser();
        tree.initByName("newick", this.createRandomNewick(), "IsLabelledNewick", true, "adjustTipHeights", false);

        TreeIntervals expected = new TreeIntervals();
        expected.initByName("tree", tree);
        actual.initByName("tree", tree);

        this.assertSameIntervals(expected, actual);

        for (int move = 0; move < NUM_MOVES; move++) {
            Node node = tree.getNode(NUM_TIPS + Randomizer.nextInt(NUM_TIPS - 1));
            double lower = Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
            double upper = node.isRoot() ? lower + 1.0 : node.getParent().getHeight();
            node.setHeight(lower + Randomizer.nextDouble() * (upper - lower));

            expected.setIntervalsUnknown();
            actual.setIntervalsUnknown();
            this.assertSameIntervals(expected, actual);
        }
    }

    private void assertSameIntervals(TreeIntervals expected, TreeIntervals actual) {
        assertEquals(expected.getIntervalCount(), actual.getIntervalCount());
        for (int i = 0; i < expected.getIntervalCount(); i++) {
            assertEquals(expected.getInterval(i), actual.getInterval(i), 1e-12);
            assertEquals(expected.getLineageCount(i), actual.getLineageCount(i));
        }
    }

    /**
     * Creates a random tree by coalescing random pairs of lineages at increasing times, with the
     * tips sampled at one of a few shared times.
     */
    private String createRandomNewick() {
        double[] sampleTimes = {0.0, 0.2, 0.5};

        List<String> lineages = new ArrayList<>();
        List<Double> heights = new ArrayList<>();
        for (int i = 0; i < NUM_TIPS; i++) {
            lineages.add("t" + i);
            heights.add(sampleTimes[Randomizer.nextInt(sampleTimes.length)]);
        }

        double time = 0.5;
        while (lineages.size() > 1) {
            time += Randomizer.nextExponential(1.0);
            int a = Randomizer.nextInt(lineages.size());
            String left = lineages.remove(a) + ":" + (time - heights.remove(a));
            int b = Randomizer.nextInt(lineages.size());
            String right = lineages.remove(b) + ":" + (time - heights.remove(b));
            lineages.add("(" + left + "," + right + ")");
            heights.add(time);
        }
        return lineages.get(0) + ";";
    }
}
