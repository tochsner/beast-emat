package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import emat.diagnostics.DiagnosedOperator;
import emat.diagnostics.StepAnnotations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Description("Tilts the intervals between every internal node and its older child by a factor that depends on the " +
        "position of the node in the tree, and rescales all of them by a common factor so that the tree length stays " +
        "the same. This redistributes the intervals between the young and the old part of the tree without changing " +
        "the total length. The older child of a node is the higher of its new children, so it can change. The mutations are " +
        "mapped linearly within their branches.")
public class TiltedIntervalScaleOperator extends ScaleTunedOperator implements DiagnosedOperator {

    /** The position of a node that the tilt depends on. */
    public enum Coordinate {
        /** The number of edges between the node and the root, which no proposal changes. */
        DEPTH,
        /** The rank of the height of the node among all internal nodes, whose order a proposal must keep. */
        RANK
    }

    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the tilt scale factor", 10.0);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the tilt scale factor", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "initial standard deviation of the log factor of the tilt", 0.03);
    final public Input<Coordinate> coordinateInput = new Input<>("coordinate", "the position of a node that the tilt depends on", Coordinate.DEPTH, Coordinate.values());

    private static final int NUM_BISECTION_STEPS = 55;

    private Coordinate coordinate;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.scaleFactor = this.scaleFactorInput.get();
        this.upper = this.scaleUpperLimit.get();
        this.lower = this.scaleLowerLimit.get();
        this.coordinate = this.coordinateInput.get();
    }

    /**
     * Multiplies the interval w_u of every internal node u, which is the distance to its higher child, by exp(a g_u - c)
     * for a standardised coordinate g of the nodes with zero sum and a tilt a, and rebuilds the heights from the
     * youngest nodes up, so that the older child of a node is the higher one of its new children. The intervals are
     * coordinates of the tree for the fixed topology and tip heights, and the new tree has exactly the scaled intervals as
     * coordinates, which makes the move a bijection with the reverse tilt -a. The constant c keeps the tree length the same.
     * The tree length increases in the intervals, so c is unique and the reverse move finds the same c. The tree length is
     * a constant of the tips that end the chains of older children plus Σ q_u w_u, with weights q_u that count the nodes
     * whose chain contains the node. Differentiating the length condition shows that the determinant of the map from the
     * intervals is the product of the factors times Λ / Λ', where Λ and Λ' are the sums Σ q_u w_u before and after. Both
     * sums are the same unless the older child of a node changes. With Σ g_u = 0 the product of the factors is exp(-n c).
     */
    @Override
    public double proposal() {
        double tilt = this.drawTilt();

        Node[] nodes = this.tree.getNodesAsArray();
        double[] oldHeights = new double[nodes.length];
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
        }

        Node[] internalNodes = this.collectInternalNodesFromRoot();
        int numInternalNodes = internalNodes.length;
        int[] indexOfNode = new int[nodes.length];
        for (int i = 0; i < numInternalNodes; i++) {
            indexOfNode[internalNodes[i].getNr()] = i;
        }

        double[] coordinates = this.computeCoordinates(internalNodes, indexOfNode);
        if (coordinates == null) {
            return Double.NEGATIVE_INFINITY;
        }

        double[] widths = new double[numInternalNodes];
        for (int i = 0; i < numInternalNodes; i++) {
            Node node = internalNodes[i];
            widths[i] = node.getHeight() - Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
        }

        double targetLength = NodeTimes.computeTreeLength(this.tree);
        double logNormaliser = this.findLogNormaliser(internalNodes, widths, coordinates, tilt, oldHeights, targetLength);
        if (!Double.isFinite(logNormaliser)) {
            return Double.NEGATIVE_INFINITY;
        }

        double[] newHeights = this.computeHeights(internalNodes, widths, coordinates, tilt, logNormaliser, oldHeights);
        if (this.coordinate == Coordinate.RANK && !this.isRankOrderKept(internalNodes, newHeights)) {
            return Double.NEGATIVE_INFINITY;
        }

        for (Node node : internalNodes) {
            node.setHeight(newHeights[node.getNr()]);
        }

        StepAnnotations.set(0, tilt);
        StepAnnotations.set(1, logNormaliser);
        StepAnnotations.set(2, numInternalNodes);

        double lengthSumBefore = this.computeWeightedIntervalSum(internalNodes, indexOfNode, oldHeights);
        double lengthSumAfter = this.computeWeightedIntervalSum(internalNodes, indexOfNode, newHeights);

        double logHastingsRatio = -numInternalNodes * logNormaliser + Math.log(lengthSumBefore / lengthSumAfter);
        logHastingsRatio += BranchTimeMapper.mapMutations(this.mutations, nodes, oldHeights, this);

        return logHastingsRatio;
    }

    /**
     * Finds by bisection the log normaliser c for which the tilted intervals give the target tree length. The factors
     * exp(a g - c) are all at most one for c = max(a g) and all at least one for c = min(a g), which brackets c.
     */
    private double findLogNormaliser(Node[] internalNodes, double[] widths, double[] coordinates, double tilt, double[] oldHeights, double targetLength) {
        double minShift = Arrays.stream(coordinates).map(value -> tilt * value).min().orElse(0.0);
        double maxShift = Arrays.stream(coordinates).map(value -> tilt * value).max().orElse(0.0);

        double low = minShift;
        double high = maxShift;
        for (int i = 0; i < NUM_BISECTION_STEPS; i++) {
            double middle = 0.5 * (low + high);
            double length = this.computeTreeLength(this.computeHeights(internalNodes, widths, coordinates, tilt, middle, oldHeights));

            // the tree length decreases in the log normaliser
            if (length > targetLength) {
                low = middle;
            } else {
                high = middle;
            }
        }

        return 0.5 * (low + high);
    }

    /** Returns the heights by node number after the tilt, with every internal node placed above its higher new child. */
    private double[] computeHeights(Node[] internalNodes, double[] widths, double[] coordinates, double tilt, double logNormaliser, double[] oldHeights) {
        double[] newHeights = oldHeights.clone();
        for (int i = internalNodes.length - 1; i >= 0; i--) {
            Node node = internalNodes[i];
            newHeights[node.getNr()] = Math.max(newHeights[node.getLeft().getNr()], newHeights[node.getRight().getNr()])
                    + widths[i] * Math.exp(tilt * coordinates[i] - logNormaliser);
        }
        return newHeights;
    }

    private double computeTreeLength(double[] heights) {
        double length = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                length += heights[node.getParent().getNr()] - heights[node.getNr()];
            }
        }
        return length;
    }

    /** Draws the tilt, whose distribution is symmetric around zero. */
    protected double drawTilt() {
        return Math.log(this.kernelDistribution.getScaler(0, this.getCoercableParameterValue()));
    }

    @Override
    public String[] getAnnotationNames() {
        return new String[]{"tilt", "logNormaliser", "numInternalNodes"};
    }

    /** Returns the internal nodes with every node placed after its parent, starting with the root. */
    private Node[] collectInternalNodesFromRoot() {
        List<Node> internalNodes = new ArrayList<>();
        List<Node> stack = new ArrayList<>();
        stack.add(this.tree.getRoot());

        while (!stack.isEmpty()) {
            Node node = stack.removeLast();
            if (node.isLeaf()) {
                continue;
            }
            internalNodes.add(node);
            stack.add(node.getLeft());
            stack.add(node.getRight());
        }

        return internalNodes.toArray(new Node[0]);
    }

    /**
     * Returns the standardised coordinate of every internal node, with zero mean and unit variance, or null if the
     * coordinate is the same for all nodes. A positive tilt gives more weight to the intervals of the nodes near
     * the root (the negative depth) or of the old nodes (the rank).
     */
    private double[] computeCoordinates(Node[] internalNodes, int[] indexOfNode) {
        int numInternalNodes = internalNodes.length;
        double[] values = new double[numInternalNodes];

        if (this.coordinate == Coordinate.DEPTH) {
            for (int i = 1; i < numInternalNodes; i++) {
                values[i] = values[indexOfNode[internalNodes[i].getParent().getNr()]] - 1.0;
            }
        } else {
            Integer[] order = this.sortByHeight(internalNodes, null);
            for (int rank = 0; rank < numInternalNodes; rank++) {
                values[order[rank]] = rank;
            }
        }

        double mean = Arrays.stream(values).average().orElse(0.0);
        double variance = Arrays.stream(values).map(value -> (value - mean) * (value - mean)).average().orElse(0.0);
        if (numInternalNodes < 2 || variance < 1e-12) {
            return null;
        }

        double standardDeviation = Math.sqrt(variance);
        for (int i = 0; i < numInternalNodes; i++) {
            values[i] = (values[i] - mean) / standardDeviation;
        }
        return values;
    }

    /**
     * Returns Σ q_u w_u over the internal nodes for the given heights by node number, where w_u is the distance of node u to
     * its higher child and q_u is the number of internal nodes whose chain of higher children contains u, plus one if u is
     * on the chain of the root. With the tips fixed, the tree length is this sum plus a constant of the tips that end the chains.
     */
    private double computeWeightedIntervalSum(Node[] internalNodes, int[] indexOfNode, double[] heights) {
        int numInternalNodes = internalNodes.length;
        int[] chainCounts = new int[numInternalNodes];
        boolean[] isOnRootChain = new boolean[numInternalNodes];

        double sum = 0.0;
        for (int i = 0; i < numInternalNodes; i++) {
            Node node = internalNodes[i];
            if (i == 0) {
                chainCounts[i] = 1;
                isOnRootChain[i] = true;
            } else {
                int parentIndex = indexOfNode[node.getParent().getNr()];
                Node parent = internalNodes[parentIndex];
                Node olderChild = heights[parent.getLeft().getNr()] >= heights[parent.getRight().getNr()] ? parent.getLeft() : parent.getRight();
                boolean isOlderChild = olderChild == node;

                chainCounts[i] = 1 + (isOlderChild ? chainCounts[parentIndex] : 0);
                isOnRootChain[i] = isOlderChild && isOnRootChain[parentIndex];
            }

            double interval = heights[node.getNr()] - Math.max(heights[node.getLeft().getNr()], heights[node.getRight().getNr()]);
            sum += (chainCounts[i] + (isOnRootChain[i] ? 1 : 0)) * interval;
        }
        return sum;
    }

    /** Returns whether the new heights order the internal nodes like the current heights do. */
    private boolean isRankOrderKept(Node[] internalNodes, double[] newHeights) {
        Integer[] oldOrder = this.sortByHeight(internalNodes, null);
        Integer[] newOrder = this.sortByHeight(internalNodes, newHeights);
        return Arrays.equals(oldOrder, newOrder);
    }

    /** Returns the indices of the internal nodes sorted by height, which is read from the nodes if no heights are given by node number. */
    private Integer[] sortByHeight(Node[] internalNodes, double[] heightsByNodeNr) {
        Integer[] order = new Integer[internalNodes.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(
                heightsByNodeNr == null ? internalNodes[a].getHeight() : heightsByNodeNr[internalNodes[a].getNr()],
                heightsByNodeNr == null ? internalNodes[b].getHeight() : heightsByNodeNr[internalNodes[b].getNr()]));
        return order;
    }

}
