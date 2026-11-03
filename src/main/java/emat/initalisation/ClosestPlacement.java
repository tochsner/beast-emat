package emat.initalisation;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.TraitSet;
import beast.base.evolution.tree.Tree;

import java.util.Comparator;
import java.util.stream.IntStream;

/**
 * Places the tips of a tree one at a time, oldest first, each next to the closest tip placed
 * before it. The closest tip is the one with the fewest differences at the variable sites, and
 * ties go to the tip placed first. Every tip is compared with all tips placed before it, but the
 * counting of differences stops once a candidate cannot be closer than the best one so far, and
 * the tips are searched in parallel. Tip i is the taxon i of the tree.
 */
public class ClosestPlacement {

    private final double[] tipHeights;
    private final PackedSequences sequences;

    // the tips in the order they are placed, and per step the closest tip and its differences
    private final int[] order;
    private final int[] closestTips;
    private final int[] numDifferences;

    public ClosestPlacement(Tree tree, Alignment alignment) {
        String[] taxaNames = tree.getTaxaNames();
        this.tipHeights = this.computeTipHeights(tree, taxaNames);
        this.sequences = new PackedSequences(alignment, taxaNames);

        this.order = this.sortOldestFirst();
        this.closestTips = new int[this.order.length];
        this.numDifferences = new int[this.order.length];
        IntStream.range(1, this.order.length).parallel().forEach(this::findClosestTip);
    }

    /**
     * Returns the height of every tip, i.e. the time before the most recent tip. Without a date
     * trait all tips have height zero.
     */
    private double[] computeTipHeights(Tree tree, String[] taxaNames) {
        double[] tipHeights = new double[taxaNames.length];
        TraitSet dateTrait = tree.getDateTrait();
        if (dateTrait == null) {
            return tipHeights;
        }

        // the trait of a date stores the height before the most recent sample
        for (int i = 0; i < taxaNames.length; i++) {
            tipHeights[i] = dateTrait.getValue(taxaNames[i]);
        }
        return tipHeights;
    }

    /** Returns the tips sorted by decreasing height, with ties broken by tip number. */
    private int[] sortOldestFirst() {
        return IntStream.range(0, this.tipHeights.length).boxed()
                .sorted(Comparator.comparingDouble((Integer tip) -> -this.tipHeights[tip]).thenComparingInt(tip -> tip))
                .mapToInt(Integer::intValue)
                .toArray();
    }

    /** Finds the closest of the tips placed before the given step to the tip placed at that step. */
    private void findClosestTip(int step) {
        int tip = this.order[step];
        int closestTip = -1;
        int closestDifferences = Integer.MAX_VALUE;

        for (int previousStep = 0; previousStep < step && closestDifferences > 0; previousStep++) {
            int candidate = this.order[previousStep];
            int candidateDifferences = this.sequences.countDifferences(tip, candidate, closestDifferences);
            if (candidateDifferences < closestDifferences) {
                closestTip = candidate;
                closestDifferences = candidateDifferences;
            }
        }

        this.closestTips[step] = closestTip;
        this.numDifferences[step] = closestDifferences;
    }

    /**
     * Returns the topology that results from inserting a new internal node on the branch above
     * the closest tip of every placed tip, with the placed tip as its other child. The internal
     * node of step k is the node numTips + k - 1.
     */
    public BinaryTopology computeTopology() {
        int numTips = this.order.length;
        int[] parents = new int[2 * numTips - 1];
        parents[this.order[0]] = -1;

        for (int step = 1; step < numTips; step++) {
            int join = numTips + step - 1;
            int closestTip = this.closestTips[step];
            parents[join] = parents[closestTip];
            parents[closestTip] = join;
            parents[this.order[step]] = join;
        }
        return new BinaryTopology(numTips, parents);
    }

    /* Accessors */

    public double[] getTipHeights() {
        return this.tipHeights;
    }

    public PackedSequences getSequences() {
        return this.sequences;
    }

    /** Returns the tips in the order they are placed. */
    public int[] getOrder() {
        return this.order;
    }

    /** Returns the closest earlier tip of the tip placed at every step, which is unused at step 0. */
    public int[] getClosestTips() {
        return this.closestTips;
    }

    /** Returns the differences of the tip placed at every step to its closest tip. */
    public int[] getNumDifferences() {
        return this.numDifferences;
    }

}
