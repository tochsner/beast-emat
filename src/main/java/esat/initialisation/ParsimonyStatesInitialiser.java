package esat.initialisation;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.StateNode;
import beast.base.inference.StateNodeInitialiser;
import beast.base.spec.inference.parameter.RealScalarParam;
import emat.helper.FitchParsimony;
import esat.state.BranchChanges;
import esat.state.NodeStates;

import java.util.ArrayList;
import java.util.List;

@Description("Initialises the node states with a Fitch parsimony reconstruction on the current tree. A node keeps " +
        "the state of its parent wherever its Fitch state set allows it, so ambiguous and missing tip characters " +
        "are filled in with the state of the parent and the initial states are compatible with the data. Unlike a " +
        "draw from the model, the reconstruction does not depend on the initial values of the parameters. Optionally, " +
        "the clock rate is set to the number of changes per site and unit of tree length, which a strict clock " +
        "needs to explain the reconstruction. Initialisers of the tree must come before this one.")
public class ParsimonyStatesInitialiser extends BEASTObject implements StateNodeInitialiser {

    final public Input<NodeStates> nodeStatesInput = new Input<>("nodeStates", "the node states to initialise", Input.Validate.REQUIRED);
    final public Input<RealScalarParam<?>> clockRateInput = new Input<>("clockRate", "the clock rate to set to the number of changes of the reconstruction per site and unit of tree length. " +
            "This rate errs on the low side once the initial tree shrinks to fit the data, which lets the branches grow, whereas a much too high rate compresses the tree");

    @Override
    public void initAndValidate() {
    }

    @Override
    public void initStateNodes() {
        NodeStates nodeStates = this.nodeStatesInput.get();
        TreeInterface tree = nodeStates.getTree();
        Alignment alignment = nodeStates.getAlignment();

        FitchParsimony fitchParsimony = new FitchParsimony(alignment, tree);
        int[][] patternStates = new int[tree.getNodeCount()][alignment.getPatternCount()];
        this.chooseStates(fitchParsimony, tree.getRoot(), null, patternStates);

        int[] rootSequence = this.expandPatternStates(alignment, patternStates[tree.getRoot().getNr()]);
        List<BranchChanges> changesAtNode = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            Node node = tree.getNode(nodeNr);
            changesAtNode.add(node.isRoot()
                    ? BranchChanges.empty(nodeNr)
                    : this.createChanges(alignment, nodeNr, patternStates[node.getParent().getNr()], patternStates[nodeNr]));
        }

        nodeStates.initialiseStates(rootSequence, changesAtNode);

        if (this.clockRateInput.get() != null) {
            this.initialiseClockRate(nodeStates, tree, alignment);
        }
    }

    /** Sets the clock rate to the number of changes per site and unit of tree length. */
    private void initialiseClockRate(NodeStates nodeStates, TreeInterface tree, Alignment alignment) {
        double treeLength = 0.0;
        for (Node node : tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                treeLength += node.getLength();
            }
        }

        // at least one change, so that the rate stays positive on data without differences
        double numChanges = Math.max(nodeStates.getDifferenceCount(), 1);
        if (treeLength > 0.0) {
            this.clockRateInput.get().set(numChanges / (alignment.getSiteCount() * treeLength));
        }
    }

    /**
     * Chooses the state of the given node and of every node below it for every pattern. A node
     * keeps the state of its parent if its state set allows it, which avoids a change, and
     * otherwise takes the lowest state of its set, as does the root. The tree is traversed
     * with an explicit stack, as deep trees could overflow the call stack.
     */
    private void chooseStates(FitchParsimony fitchParsimony, Node root, int[] rootParentStates, int[][] patternStates) {
        List<Node> stack = new ArrayList<>();
        stack.add(root);

        while (!stack.isEmpty()) {
            Node node = stack.removeLast();
            int[] nodeStates = patternStates[node.getNr()];
            int[] parentStates = node.isRoot() ? rootParentStates : patternStates[node.getParent().getNr()];

            for (int patternNr = 0; patternNr < nodeStates.length; patternNr++) {
                long stateSet = fitchParsimony.getStateSet(node, patternNr);
                boolean canKeepParentState = parentStates != null && (stateSet & (1L << parentStates[patternNr])) != 0L;
                nodeStates[patternNr] = canKeepParentState ? parentStates[patternNr] : Long.numberOfTrailingZeros(stateSet);
            }

            stack.addAll(node.getChildren());
        }
    }

    /** Returns the sequence over all sites of the given states per pattern. */
    private int[] expandPatternStates(Alignment alignment, int[] patternStates) {
        int[] sequence = new int[alignment.getSiteCount()];
        for (int site = 0; site < sequence.length; site++) {
            sequence[site] = patternStates[alignment.getPatternIndex(site)];
        }
        return sequence;
    }

    /** Returns the changes of a node with the given states per pattern relative to the given states of its parent. */
    private BranchChanges createChanges(Alignment alignment, int nodeNr, int[] parentPatternStates, int[] nodePatternStates) {
        int numSites = alignment.getSiteCount();
        int[] sites = new int[numSites];
        int[] fromStates = new int[numSites];
        int[] toStates = new int[numSites];
        int size = 0;

        for (int site = 0; site < numSites; site++) {
            int patternNr = alignment.getPatternIndex(site);
            if (parentPatternStates[patternNr] != nodePatternStates[patternNr]) {
                sites[size] = site;
                fromStates[size] = parentPatternStates[patternNr];
                toStates[size] = nodePatternStates[patternNr];
                size++;
            }
        }

        return BranchChanges.ofSorted(nodeNr, sites, fromStates, toStates, size);
    }

    @Override
    public void getInitialisedStateNodes(List<StateNode> stateNodes) {
        stateNodes.add(this.nodeStatesInput.get());
        if (this.clockRateInput.get() != null) {
            stateNodes.add(this.clockRateInput.get());
        }
    }

}
