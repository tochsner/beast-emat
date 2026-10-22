package emat.operators.topology.mdspr;

import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import emat.operators.topology.MutationDirectedSprOperator.Region;

import java.util.Arrays;

/**
 * The regions explored by one proposal, with their log weights and, once normalised,
 * their weights relative to the largest one.
 */
public class RegionBuffer {

    private static final int INITIAL_CAPACITY = 64;

    private int size;
    private Node[] nodes = new Node[INITIAL_CAPACITY];
    private int[] indices = new int[INITIAL_CAPACITY];
    private int[] numDifferences = new int[INITIAL_CAPACITY];
    private double[] lowerHeights = new double[INITIAL_CAPACITY];
    private double[] upperHeights = new double[INITIAL_CAPACITY];
    private double[] weights = new double[INITIAL_CAPACITY];
    private double totalWeight;

    public int size() {
        return this.size;
    }

    public void clear() {
        this.size = 0;
    }

    public void add(Region region, double logWeight) {
        if (this.size == this.nodes.length) {
            this.grow(2 * this.size);
        }

        int regionNr = this.size++;
        this.nodes[regionNr] = region.node();
        this.indices[regionNr] = region.index();
        this.numDifferences[regionNr] = region.numDifferences();
        this.lowerHeights[regionNr] = region.lowerHeight();
        this.upperHeights[regionNr] = region.upperHeight();
        this.weights[regionNr] = logWeight;
    }

    /** Turns the log weights into weights relative to the largest one and returns the log of their total. */
    public double normaliseWeights() {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < this.size; i++) {
            max = Math.max(max, this.weights[i]);
        }

        this.totalWeight = 0.0;
        for (int i = 0; i < this.size; i++) {
            this.weights[i] = Math.exp(this.weights[i] - max);
            this.totalWeight += this.weights[i];
        }
        return max + Math.log(this.totalWeight);
    }

    /** Samples a region with probability proportional to its normalised weight. */
    public Region sample() {
        double threshold = Randomizer.nextDouble() * this.totalWeight;
        double cumulativeWeight = 0.0;
        int lastCandidate = 0;
        for (int i = 0; i < this.size; i++) {
            if (this.weights[i] <= 0.0) {
                continue;
            }

            lastCandidate = i;
            cumulativeWeight += this.weights[i];
            if (threshold < cumulativeWeight) {
                break;
            }
        }
        return this.get(lastCandidate);
    }

    private Region get(int regionNr) {
        return new Region(
                this.nodes[regionNr], this.indices[regionNr],
                this.lowerHeights[regionNr], this.upperHeights[regionNr],
                this.numDifferences[regionNr]
        );
    }

    private void grow(int capacity) {
        this.nodes = Arrays.copyOf(this.nodes, capacity);
        this.indices = Arrays.copyOf(this.indices, capacity);
        this.numDifferences = Arrays.copyOf(this.numDifferences, capacity);
        this.lowerHeights = Arrays.copyOf(this.lowerHeights, capacity);
        this.upperHeights = Arrays.copyOf(this.upperHeights, capacity);
        this.weights = Arrays.copyOf(this.weights, capacity);
    }
}
