package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import emat.diagnostics.DiagnosedOperator;
import emat.diagnostics.StepAnnotations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

@Description("Diagnostic proposal that multiplies the interval of every internal node by exp(a g - c) for a structured " +
        "coordinate g of the node and a tilt a that is drawn uniformly, and maps the mutations linearly within their " +
        "branches. For the coordinates that are not made to keep the tree length, c is zero. The Hastings ratio is only right for " +
        "the coordinates that do not depend on the intervals (every coordinate except the rank and the log interval), and the operator is meant to be probed and never used in a chain. It draws its " +
        "random numbers from its own generator.")
public class TiltProbeOperator extends ScaleTunedOperator implements DiagnosedOperator {

    /** The property of a node that the tilt depends on. */
    public enum Coordinate {
        /** Minus the number of edges between the node and the root. */
        DEPTH,
        /** The square of the depth, made orthogonal to a constant and to the depth over the internal nodes. */
        DEPTH_CURVATURE,
        /** The logarithm of the number of tips below the node. */
        LOG_CLADE_SIZE,
        /** The rank of the height of the node among the internal nodes. */
        HEIGHT_RANK,
        /** The logarithm of the interval of the node. */
        LOG_INTERVAL,
        /** One for the nodes above a quantile of the heights and zero below, without a constant tree length. */
        OLD_STEP,
        /** One for the given number of oldest nodes and zero elsewhere, without a constant tree length. */
        TOP_COUNT,
        /** Zero up to a quantile of the heights, then linear in the rank up to one at the root, without a constant tree length. */
        OLD_RAMP,
        /** The height of the node relative to the height of the root, without a constant tree length. */
        AGE,
        /** The exponential of minus the depth divided by a decay length, without a constant tree length. */
        ROOT_DECAY,
        /** One for every node plus a weight on the root, without a constant tree length. */
        UNIFORM_ROOT,
        /** Minus the logarithm of one plus the mutations on the branch above the older child. */
        MUTATION_SPARSE,
        /** A given value per height rank. */
        RANK_PROFILE,
        /** A given value per height rank, without a constant tree length. */
        RANK_PROFILE_RAW,
        /** One for the nodes with two tips as children and zero elsewhere, without a constant tree length. */
        CHERRY,
        /** Plus one below the left child of the root, minus one below the right child. */
        ROOT_SPLIT
    }

    final public Input<Coordinate> coordinateInput = new Input<>("coordinate", "the property of a node that the tilt depends on", Coordinate.DEPTH, Coordinate.values());
    final public Input<Double> maxTiltInput = new Input<>("maxTilt", "the tilt is drawn uniformly between minus and plus this value", 0.1);
    final public Input<Double> quantileInput = new Input<>("quantile", "the quantile of the node heights above which the step coordinate is one", 0.5);
    final public Input<Integer> topCountInput = new Input<>("topCount", "the number of oldest nodes for which the top count coordinate is one", 1);
    final public Input<Double> decayInput = new Input<>("decay", "the decay length in edges of the root decay coordinate", 2.0);
    final public Input<Double> rootWeightInput = new Input<>("rootWeight", "the weight added on the root by the uniform root coordinate", 5.0);
    final public Input<String> rankProfileInput = new Input<>("rankProfile", "comma separated values of the rank profile coordinates, from the youngest to the oldest node", "");
    final public Input<Long> seedInput = new Input<>("seed", "seed of the generator of the tilts", 12345L);

    private Random random;
    private double[] rankProfile;

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        this.random = new Random(this.seedInput.get());
        this.rankProfile = this.rankProfileInput.get().isBlank()
                ? new double[0]
                : Arrays.stream(this.rankProfileInput.get().split(",")).mapToDouble(Double::parseDouble).toArray();
    }

    @Override
    public double proposal() {
        Node[] nodes = this.tree.getNodesAsArray();
        double[] oldHeights = new double[nodes.length];
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
        }

        Node[] internalNodes = this.collectInternalNodesFromYoungest();
        double[] intervals = new double[nodes.length];
        for (Node node : internalNodes) {
            intervals[node.getNr()] = oldHeights[node.getNr()] - Math.max(oldHeights[node.getLeft().getNr()], oldHeights[node.getRight().getNr()]);
        }

        double[] coordinates = this.computeCoordinates(internalNodes, oldHeights, intervals);
        if (coordinates == null) {
            return Double.NEGATIVE_INFINITY;
        }

        double tilt = (2.0 * this.random.nextDouble() - 1.0) * this.maxTiltInput.get();
        double lengthBefore = NodeTimes.computeTreeLength(this.tree);

        double normaliser = this.hasConstantTreeLength()
                ? this.findNormaliser(internalNodes, intervals, coordinates, tilt, lengthBefore)
                : 0.0;
        if (Double.isNaN(normaliser)) {
            return Double.NEGATIVE_INFINITY;
        }

        this.setHeights(internalNodes, intervals, coordinates, tilt, normaliser);
        double logIntervalJacobian = this.computeLogIntervalJacobian(internalNodes, coordinates, tilt, normaliser);

        double logMutationJacobian;
        this.mutations.setSanityCheckingEnabled(false);
        try {
            logMutationJacobian = BranchTimeMapper.mapMutations(this.mutations, nodes, oldHeights, this);
        } finally {
            this.mutations.setSanityCheckingEnabled(true);
        }

        StepAnnotations.set(0, tilt);
        StepAnnotations.set(1, Math.log(NodeTimes.computeTreeLength(this.tree) / lengthBefore));
        StepAnnotations.set(2, normaliser);
        StepAnnotations.set(3, logIntervalJacobian);
        StepAnnotations.set(4, logMutationJacobian);

        return logIntervalJacobian + logMutationJacobian;
    }

    @Override
    public String[] getAnnotationNames() {
        return new String[]{"tilt", "logTreeLengthRatio", "normaliser", "logIntervalJacobian", "logMutationJacobian"};
    }

    /**
     * Returns the log Jacobian of the map of the intervals. The factor of every interval does not depend on the
     * intervals except through the constant, which is homogeneous of degree zero, so the determinant is the product
     * of the factors. This holds for the coordinates that do not depend on the intervals, the depth and the step.
     */
    private double computeLogIntervalJacobian(Node[] internalNodes, double[] coordinates, double tilt, double normaliser) {
        if (this.coordinateInput.get() == Coordinate.HEIGHT_RANK || this.coordinateInput.get() == Coordinate.LOG_INTERVAL) {
            return Double.NaN;
        }

        double sum = 0.0;
        for (Node node : internalNodes) {
            sum += tilt * coordinates[node.getNr()] - normaliser;
        }
        return sum;
    }

    /** Returns the internal nodes with every node placed after its children. */
    private Node[] collectInternalNodesFromYoungest() {
        List<Node> nodes = new ArrayList<>();
        List<Node> stack = new ArrayList<>();
        stack.add(this.tree.getRoot());

        while (!stack.isEmpty()) {
            Node node = stack.removeLast();
            if (!node.isLeaf()) {
                nodes.add(node);
                stack.add(node.getLeft());
                stack.add(node.getRight());
            }
        }

        Collections.reverse(nodes);
        return nodes.toArray(new Node[0]);
    }

    /** Returns the coordinate of every internal node by node number, standardised to zero mean and unit variance, or null if it is constant. */
    private double[] computeCoordinates(Node[] internalNodes, double[] oldHeights, double[] intervals) {
        double[] values = new double[oldHeights.length];

        switch (this.coordinateInput.get()) {
            case DEPTH -> {
                for (int i = internalNodes.length - 1; i >= 0; i--) {
                    Node node = internalNodes[i];
                    values[node.getNr()] = node.isRoot() ? 0.0 : values[node.getParent().getNr()] - 1.0;
                }
            }
            case DEPTH_CURVATURE -> {
                double[] depths = new double[values.length];
                for (int i = internalNodes.length - 1; i >= 0; i--) {
                    Node node = internalNodes[i];
                    depths[node.getNr()] = node.isRoot() ? 0.0 : depths[node.getParent().getNr()] - 1.0;
                }
                this.fillOrthogonalSquare(internalNodes, depths, values);
            }
            case LOG_CLADE_SIZE -> {
                for (Node node : internalNodes) {
                    values[node.getNr()] = Math.log(node.getLeafNodeCount());
                }
            }
            case HEIGHT_RANK, OLD_STEP, TOP_COUNT, OLD_RAMP, RANK_PROFILE, RANK_PROFILE_RAW -> {
                Node[] byHeight = Arrays.stream(internalNodes).sorted((a, b) -> Double.compare(oldHeights[a.getNr()], oldHeights[b.getNr()])).toArray(Node[]::new);
                for (int rank = 0; rank < byHeight.length; rank++) {
                    values[byHeight[rank].getNr()] = this.computeRankValue(rank, byHeight.length);
                }
            }
            case AGE -> {
                double rootHeight = this.tree.getRoot().getHeight();
                for (Node node : internalNodes) {
                    values[node.getNr()] = oldHeights[node.getNr()] / rootHeight;
                }
            }
            case ROOT_DECAY -> {
                for (int i = internalNodes.length - 1; i >= 0; i--) {
                    Node node = internalNodes[i];
                    values[node.getNr()] = node.isRoot() ? 1.0 : values[node.getParent().getNr()] * Math.exp(-1.0 / this.decayInput.get());
                }
            }
            case UNIFORM_ROOT -> {
                for (Node node : internalNodes) {
                    values[node.getNr()] = node.isRoot() ? 1.0 + this.rootWeightInput.get() : 1.0;
                }
            }
            case MUTATION_SPARSE -> {
                for (Node node : internalNodes) {
                    Node older = oldHeights[node.getLeft().getNr()] >= oldHeights[node.getRight().getNr()] ? node.getLeft() : node.getRight();
                    values[node.getNr()] = -Math.log(1.0 + this.mutations.getMutations(older).size());
                }
            }
            case CHERRY -> {
                for (Node node : internalNodes) {
                    values[node.getNr()] = node.getLeft().isLeaf() && node.getRight().isLeaf() ? 1.0 : 0.0;
                }
            }
            case ROOT_SPLIT -> {
                for (int i = internalNodes.length - 1; i >= 0; i--) {
                    Node node = internalNodes[i];
                    if (!node.isRoot()) {
                        values[node.getNr()] = node.getParent().isRoot()
                                ? (node == node.getParent().getLeft() ? 1.0 : -1.0)
                                : values[node.getParent().getNr()];
                    }
                }
            }
            case LOG_INTERVAL -> {
                for (Node node : internalNodes) {
                    values[node.getNr()] = Math.log(Math.max(intervals[node.getNr()], 1e-12));
                }
            }
        }

        if (!this.hasConstantTreeLength()) {
            return values;
        }

        double mean = Arrays.stream(internalNodes).mapToDouble(node -> values[node.getNr()]).average().orElse(0.0);
        double variance = Arrays.stream(internalNodes).mapToDouble(node -> (values[node.getNr()] - mean) * (values[node.getNr()] - mean)).average().orElse(0.0);
        if (variance < 1e-12) {
            return null;
        }

        for (Node node : internalNodes) {
            values[node.getNr()] = (values[node.getNr()] - mean) / Math.sqrt(variance);
        }
        return values;
    }

    /** Returns whether the tilt is followed by a constant that keeps the tree length the same. */
    private boolean hasConstantTreeLength() {
        return switch (this.coordinateInput.get()) {
            case OLD_STEP, TOP_COUNT, OLD_RAMP, AGE, ROOT_DECAY, UNIFORM_ROOT, RANK_PROFILE_RAW, CHERRY -> false;
            default -> true;
        };
    }

    /** Returns the value of the rank coordinates for a node with the given rank of its height. */
    private double computeRankValue(int rank, int count) {
        return switch (this.coordinateInput.get()) {
            case OLD_STEP -> rank >= Math.floor(this.quantileInput.get() * count) ? 1.0 : 0.0;
            case TOP_COUNT -> rank >= count - this.topCountInput.get() ? 1.0 : 0.0;
            case OLD_RAMP -> {
                double first = Math.floor(this.quantileInput.get() * count);
                yield rank >= first ? (rank - first) / Math.max(1.0, count - 1 - first) : 0.0;
            }
            case RANK_PROFILE, RANK_PROFILE_RAW -> this.rankProfile[Math.min(rank, this.rankProfile.length - 1)];
            default -> rank;
        };
    }

    /** Fills the squares of the given values, after removing their projection on a constant and on the values themselves. */
    private void fillOrthogonalSquare(Node[] internalNodes, double[] depths, double[] result) {
        int n = internalNodes.length;
        double meanDepth = Arrays.stream(internalNodes).mapToDouble(node -> depths[node.getNr()]).average().orElse(0.0);
        double meanSquare = Arrays.stream(internalNodes).mapToDouble(node -> depths[node.getNr()] * depths[node.getNr()]).average().orElse(0.0);

        double covariance = 0.0;
        double variance = 0.0;
        for (Node node : internalNodes) {
            double centred = depths[node.getNr()] - meanDepth;
            covariance += centred * (depths[node.getNr()] * depths[node.getNr()] - meanSquare);
            variance += centred * centred;
        }
        double slope = variance > 0.0 ? covariance / variance : 0.0;

        for (Node node : internalNodes) {
            double depth = depths[node.getNr()];
            result[node.getNr()] = depth * depth - meanSquare - slope * (depth - meanDepth);
        }
    }

    /** Finds by bisection the constant c for which the tilted intervals give the old tree length, or NaN if there is none. */
    private double findNormaliser(Node[] internalNodes, double[] intervals, double[] coordinates, double tilt, double targetLength) {
        double low = -1.0;
        double high = 1.0;

        // the tree length decreases in c
        if (this.computeTreeLength(internalNodes, intervals, coordinates, tilt, low) < targetLength
                || this.computeTreeLength(internalNodes, intervals, coordinates, tilt, high) > targetLength) {
            return Double.NaN;
        }

        for (int i = 0; i < 60; i++) {
            double middle = 0.5 * (low + high);
            if (this.computeTreeLength(internalNodes, intervals, coordinates, tilt, middle) > targetLength) {
                low = middle;
            } else {
                high = middle;
            }
        }

        return 0.5 * (low + high);
    }

    /** Returns the tree length after the tilt, with every node placed above its higher child. */
    private double computeTreeLength(Node[] internalNodes, double[] intervals, double[] coordinates, double tilt, double normaliser) {
        double[] heights = new double[this.tree.getNodeCount()];
        for (Node node : this.tree.getNodesAsArray()) {
            heights[node.getNr()] = node.getHeight();
        }

        for (Node node : internalNodes) {
            heights[node.getNr()] = Math.max(heights[node.getLeft().getNr()], heights[node.getRight().getNr()])
                    + intervals[node.getNr()] * Math.exp(tilt * coordinates[node.getNr()] - normaliser);
        }

        double length = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                length += heights[node.getParent().getNr()] - heights[node.getNr()];
            }
        }
        return length;
    }

    private void setHeights(Node[] internalNodes, double[] intervals, double[] coordinates, double tilt, double normaliser) {
        for (Node node : internalNodes) {
            node.setHeight(Math.max(node.getLeft().getHeight(), node.getRight().getHeight())
                    + intervals[node.getNr()] * Math.exp(tilt * coordinates[node.getNr()] - normaliser));
        }
    }

}
