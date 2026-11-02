package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import emat.diagnostics.DiagnosedOperator;
import emat.diagnostics.StepAnnotations;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Description("Scales the intervals between the nodes of one depth band and their older children by a common factor, " +
        "and leaves the intervals of all other nodes as they are. The internal nodes are split into bands by the quantiles " +
        "of their depth, which is the number of edges to the root and does not change with the node heights. A band is " +
        "drawn uniformly among the bands with nodes, and every band has its own scale factor that is tuned by its own " +
        "acceptance. The older child of a node is the higher of its new children, so it can change. The mutations are " +
        "mapped linearly within their branches.")
public class BandedIntervalScaleOperator extends ScaleTunedOperator implements DiagnosedOperator {

    final public Input<Integer> numBandsInput = new Input<>("numBands", "the number of depth bands", 4);
    final public Input<Double> scaleUpperLimit = new Input<>("upper", "upper limit of the scale factor of a band", 10.0);
    final public Input<Double> scaleLowerLimit = new Input<>("lower", "lower limit of the scale factor of a band", 1e-8);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "initial standard deviation of the log factor of every band", 0.03);

    protected int numBands;
    private double[] scaleFactors;
    private int lastBand;
    private int[] numAccepted;
    private int[] numProposed;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.numBands = this.numBandsInput.get();
        this.upper = this.scaleUpperLimit.get();
        this.lower = this.scaleLowerLimit.get();
        this.scaleFactors = new double[this.numBands];
        Arrays.fill(this.scaleFactors, this.scaleFactorInput.get());
        this.numAccepted = new int[this.numBands];
        this.numProposed = new int[this.numBands];
    }

    /**
     * Multiplies the interval of every internal node in a randomly drawn band by exp(a), where a is the log of a factor
     * from the kernel, and rebuilds the heights from the youngest nodes up, so that the older child of a node is the higher
     * one of its new children. The intervals are coordinates of the tree for the fixed topology and tip heights, and the new
     * tree has exactly the scaled intervals as coordinates, which makes the move a bijection with the reverse factor exp(-a).
     * The bands depend on the topology only, so the reverse move draws the same band, and the log Jacobian is a times the
     * number of nodes in the band.
     */
    @Override
    public double proposal() {
        Node[] nodes = this.tree.getNodesAsArray();
        double[] oldHeights = new double[nodes.length];
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
        }

        Node[] internalNodes = this.collectInternalNodesFromRoot();
        int[] bandOfNode = this.assignBands(internalNodes);

        int band = this.drawBand(bandOfNode);
        this.lastBand = band;
        this.numProposed[band]++;

        double logScaler = Math.log(this.kernelDistribution.getScaler(0, this.scaleFactors[band]));

        int numScaledNodes = 0;
        double[] newHeights = oldHeights.clone();
        for (int i = internalNodes.length - 1; i >= 0; i--) {
            Node node = internalNodes[i];
            double oldInterval = oldHeights[node.getNr()] - Math.max(oldHeights[node.getLeft().getNr()], oldHeights[node.getRight().getNr()]);
            boolean isScaled = bandOfNode[i] == band;
            numScaledNodes += isScaled ? 1 : 0;

            newHeights[node.getNr()] = Math.max(newHeights[node.getLeft().getNr()], newHeights[node.getRight().getNr()])
                    + oldInterval * (isScaled ? Math.exp(logScaler) : 1.0);
        }
        for (Node node : internalNodes) {
            node.setHeight(newHeights[node.getNr()]);
        }

        StepAnnotations.set(0, logScaler);
        StepAnnotations.set(1, band);
        StepAnnotations.set(2, numScaledNodes);

        double logHastingsRatio = numScaledNodes * logScaler;
        logHastingsRatio += BranchTimeMapper.mapMutations(this.mutations, nodes, oldHeights, this);

        return logHastingsRatio;
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
     * Returns the band of every internal node by its index, which is the number of the quantiles of the depths that are at
     * most the depth of the node. Band 0 holds the nodes closest to the root. Nodes with the same depth share a band.
     */
    protected int[] assignBands(Node[] internalNodes) {
        int numInternalNodes = internalNodes.length;
        int[] depths = new int[this.tree.getNodeCount()];
        int[] depthOfNode = new int[numInternalNodes];
        for (int i = 1; i < numInternalNodes; i++) {
            depths[internalNodes[i].getNr()] = depths[internalNodes[i].getParent().getNr()] + 1;
        }
        for (int i = 0; i < numInternalNodes; i++) {
            depthOfNode[i] = depths[internalNodes[i].getNr()];
        }

        int[] sortedDepths = depthOfNode.clone();
        Arrays.sort(sortedDepths);
        int[] quantiles = new int[this.numBands - 1];
        for (int k = 1; k < this.numBands; k++) {
            quantiles[k - 1] = sortedDepths[k * numInternalNodes / this.numBands];
        }

        int[] bandOfNode = new int[numInternalNodes];
        for (int i = 0; i < numInternalNodes; i++) {
            for (int quantile : quantiles) {
                bandOfNode[i] += quantile <= depthOfNode[i] ? 1 : 0;
            }
        }
        return bandOfNode;
    }

    /** Draws uniformly among the bands that hold at least one node. */
    private int drawBand(int[] bandOfNode) {
        boolean[] hasNodes = new boolean[this.numBands];
        for (int band : bandOfNode) {
            hasNodes[band] = true;
        }

        int numNonEmptyBands = 0;
        for (boolean isNonEmpty : hasNodes) {
            numNonEmptyBands += isNonEmpty ? 1 : 0;
        }

        int draw = Randomizer.nextInt(numNonEmptyBands);
        for (int band = 0; band < this.numBands; band++) {
            if (hasNodes[band] && draw-- == 0) {
                return band;
            }
        }
        throw new IllegalStateException("No band was drawn.");
    }

    @Override
    public String[] getAnnotationNames() {
        return new String[]{"logScaler", "band", "numScaledNodes"};
    }

    /* Every band is tuned by its own acceptance. */

    @Override
    public void accept() {
        super.accept();
        this.numAccepted[this.lastBand]++;
    }

    @Override
    public void optimize(double logAlpha) {
        if (this.optimiseInput.get()) {
            double delta = this.calcDelta(logAlpha) + Math.log(this.scaleFactors[this.lastBand]);
            this.scaleFactors[this.lastBand] = Math.max(Math.min(Math.exp(delta), this.upper), this.lower);
        }
    }

    /** Returns the mean of the scale factors of the bands. */
    @Override
    public double getCoercableParameterValue() {
        return Arrays.stream(this.scaleFactors).average().orElse(0.0);
    }

    @Override
    public void setCoercableParameterValue(double value) {
        Arrays.fill(this.scaleFactors, Math.max(Math.min(value, this.upper), this.lower));
    }

    /** Returns the label that introduces the bands in the performance suggestion. */
    protected String getBandDescription() {
        return "bands (scale factor, acceptance):";
    }

    /** Returns the scale factor and the acceptance probability of every band, from the closest to the root to the deepest. */
    @Override
    public String getPerformanceSuggestion() {
        DecimalFormat formatter = new DecimalFormat("#.###");
        StringBuilder suggestion = new StringBuilder(this.getBandDescription());
        for (int band = 0; band < this.numBands; band++) {
            double acceptance = this.numProposed[band] == 0 ? 0.0 : this.numAccepted[band] / (double) this.numProposed[band];
            suggestion.append(" ").append(formatter.format(this.scaleFactors[band])).append(" (").append(formatter.format(acceptance)).append(")");
        }
        return suggestion.toString();
    }

}
