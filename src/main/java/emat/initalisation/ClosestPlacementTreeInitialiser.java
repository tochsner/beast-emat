package emat.initalisation;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.inference.StateNodeInitialiser;
import beast.base.spec.inference.parameter.RealScalarParam;

import java.util.List;

@Description("Initialises a tree by placing the tips one at a time, oldest first. Each tip is attached to the " +
        "branch above the closest tip placed so far, at the height implied by the Jukes-Cantor distance and the " +
        "tip dates.")
public class ClosestPlacementTreeInitialiser extends BEASTObject implements StateNodeInitialiser {

    final public Input<Tree> treeInput = new Input<>("tree", "the tree to initialise", Input.Validate.REQUIRED);
    final public Input<Alignment> alignmentInput = new Input<>("alignment", "alignment used for the distances between tips", Input.Validate.REQUIRED);
    final public Input<RealScalarParam<?>> clockRateInput = new Input<>("clockRate", "the clock rate, in substitutions per site per time unit, used to convert distances to times (default 1.0)");

    // the smallest branch length of the starting tree, in time units, so that transition probabilities are accurate
    static final double MIN_BRANCH_LENGTH = 1e-3;

    // the largest proportion of differences the Jukes-Cantor correction is applied to, below its saturation at 0.75
    static final double MAX_P_DISTANCE = 0.749;

    @Override
    public void initAndValidate() {
    }

    /**
     * Builds the tree from the tips in order of decreasing height, which is oldest first, and
     * sets it on the state tree. Every new node is inserted between a tip and its parent, so the
     * root changes only when the second tip joins the first.
     */
    @Override
    public void initStateNodes() {
        Tree tree = this.treeInput.get();
        double clockRate = this.clockRateInput.get() == null ? 1.0 : this.clockRateInput.get().get();
        ClosestPlacement placement = new ClosestPlacement(tree, this.alignmentInput.get());
        Node[] tips = this.createTips(tree.getTaxaNames(), placement.getTipHeights());

        int[] order = placement.getOrder();
        for (int step = 1; step < order.length; step++) {
            Node closest = tips[placement.getClosestTips()[step]];
            double distance = this.computeDistance(placement.getNumDifferences()[step], placement.getSequences().getNumSites());
            this.insertTip(tips[order[step]], closest, distance / clockRate);
        }

        Node root = tips[order[0]];
        while (root.getParent() != null) {
            root = root.getParent();
        }
        root.labelInternalNodes(tips.length);
        tree.setRoot(root);
        tree.initArrays();
    }

    /** Creates the leaf nodes, numbered by their index in the tree. */
    Node[] createTips(String[] taxaNames, double[] tipHeights) {
        Node[] tips = new Node[taxaNames.length];
        for (int i = 0; i < taxaNames.length; i++) {
            tips[i] = new Node(taxaNames[i]);
            tips[i].setNr(i);
            tips[i].setHeight(tipHeights[i]);
        }
        return tips;
    }

    /** Returns the Jukes-Cantor distance for the given number of differences over all sites. */
    double computeDistance(int numDifferences, int numSites) {
        double p = Math.min((double) numDifferences / numSites, MAX_P_DISTANCE);
        return -0.75 * Math.log1p(-4.0 * p / 3.0);
    }

    /**
     * Inserts a new node between the closest tip and its parent, with the tip as its other child.
     * The two branches from the new node to the tips have a combined length of the given time,
     * which fixes its height, but the node is at least the minimum branch length above the older
     * of the two tips. Ancestors are raised where needed to keep the minimum branch length.
     */
    void insertTip(Node tip, Node closest, double time) {
        Node parent = closest.getParent();

        Node join = new Node();
        double height = 0.5 * (time + tip.getHeight() + closest.getHeight());
        join.setHeight(Math.max(height, Math.max(tip.getHeight(), closest.getHeight()) + MIN_BRANCH_LENGTH));
        join.setLeft(closest);
        join.setRight(tip);
        closest.setParent(join, false);
        tip.setParent(join, false);

        if (parent == null) {
            return;
        }

        join.setParent(parent, false);
        if (parent.getLeft() == closest) {
            parent.setLeft(join);
        } else {
            parent.setRight(join);
        }
        this.raiseAncestors(join);
    }

    /**
     * Raises every ancestor of the given node that is not at least MIN_BRANCH_LENGTH above its
     * child, so that every branch of the tree has at least the minimum length.
     */
    void raiseAncestors(Node node) {
        Node child = node;
        Node ancestor = node.getParent();
        while (ancestor != null && ancestor.getHeight() < child.getHeight() + MIN_BRANCH_LENGTH) {
            ancestor.setHeight(child.getHeight() + MIN_BRANCH_LENGTH);
            child = ancestor;
            ancestor = ancestor.getParent();
        }
    }

    @Override
    public void getInitialisedStateNodes(List<StateNode> stateNodes) {
        stateNodes.add(this.treeInput.get());
    }

} // class ClosestPlacementTreeInitialiser
