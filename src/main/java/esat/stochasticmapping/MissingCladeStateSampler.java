package esat.stochasticmapping;

import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import esat.state.BranchChanges;
import esat.state.NodeStates;

import java.util.ArrayList;
import java.util.List;

/**
 * Samples the states of a clade whose tips are all missing at some sites, together with the
 * state of the node it attaches to, from their exact joint conditional given the states of the
 * parent of the attach node and of its other children. The sites are independent. As the
 * rows of a transition matrix sum to one, the clade does not constrain the attach node, so
 * the state of the attach node has the weight
 * <pre>
 *     w(x) = U(a, x) · Π_c P_c(x, s_c),
 * </pre>
 * where a is the state of the parent, U the transition probabilities above the attach node or the
 * frequencies for the root, and c runs over the other children, which are in state s_c. Given
 * the attach node, the states of the clade are drawn down from it through the transition
 * probabilities of its branches. A node leaves the state of its parent at a site with probability
 * ε_a, which is small on short branches, so the sites where it leaves are found by geometric
 * skipping with the bound r = max_a ε_a, and a visited site leaves with probability ε_a / r. This
 * thinning is exact, and the cost of a branch scales with its number of changes instead of the
 * number of sites.
 * <p>
 * The sampler reads the model from the substitution process, which has to be updated before.
 */
public class MissingCladeStateSampler {

    /**
     * A sample at the given sites, which are strictly ascending: the states of the attach node,
     * the clade nodes in preorder, and for each of them the changes relative to its parent,
     * which keep the old changes at all other sites.
     */
    public record CladeSample(int[] attachStates, List<Node> cladeNodes, BranchChanges[] changesOfCladeNodes) {
    }

    private final NodeStates nodeStates;
    private final SubstitutionProcess substitutionProcess;
    private final int numStates;

    private final double[] siteWeights;

    // the transition probabilities of the prepared branch, the weights and total of leaving each state, and the bound on the latter
    private final double[] transitionProbabilities;
    private final double[] leaveWeights;
    private final double[] leaveTotals;
    private double maxLeaveProbability;
    private double logSkipProbability;

    public MissingCladeStateSampler(NodeStates nodeStates, SubstitutionProcess substitutionProcess) {
        this.nodeStates = nodeStates;
        this.substitutionProcess = substitutionProcess;
        this.numStates = nodeStates.getNumStates();
        if (substitutionProcess.getNumStates() != this.numStates) {
            throw new IllegalArgumentException("The node states have a different number of states than the substitution process.");
        }

        this.siteWeights = new double[this.numStates];
        this.transitionProbabilities = new double[this.numStates * this.numStates];
        this.leaveWeights = new double[this.numStates * this.numStates];
        this.leaveTotals = new double[this.numStates];
    }

    /** Samples the attach node and the clade below the given top at the given sites. */
    public CladeSample sample(Node top, int[] sites) {
        int[] attachStates = this.sampleAttachStates(top, sites);

        List<Node> cladeNodes = this.collectClade(top);
        BranchChanges[] changesOfCladeNodes = new BranchChanges[cladeNodes.size()];

        // the new states of the clade nodes at the sites, indexed by the number of the node
        int[][] newStatesOfNode = new int[this.nodeStates.getTree().getNodeCount()][];

        for (int i = 0; i < cladeNodes.size(); i++) {
            Node node = cladeNodes.get(i);
            int[] parentStates = node == top ? attachStates : newStatesOfNode[node.getParent().getNr()];

            this.prepareBranch(node);
            int[] states = parentStates.clone();
            this.drawChangedStates(parentStates, states);
            newStatesOfNode[node.getNr()] = states;

            changesOfCladeNodes[i] = this.mergeChanges(node, sites, parentStates, states);
        }

        return new CladeSample(attachStates, cladeNodes, changesOfCladeNodes);
    }

    /* Attach Node */

    /** Samples the states of the node above the given top at the given sites, with the clade of the top marginalised out. */
    private int[] sampleAttachStates(Node top, int[] sites) {
        Node attach = top.getParent();
        int numOtherChildren = attach.getChildCount() - 1;

        double[] upperProbabilities = new double[this.numStates * this.numStates];
        if (attach.isRoot()) {
            double[] frequencies = this.substitutionProcess.getFrequencies();
            for (int from = 0; from < this.numStates; from++) {
                System.arraycopy(frequencies, 0, upperProbabilities, from * this.numStates, this.numStates);
            }
        } else {
            this.substitutionProcess.getTransitionProbabilities(attach, upperProbabilities);
        }

        Node[] otherChildren = new Node[numOtherChildren];
        double[][] childProbabilities = new double[numOtherChildren][this.numStates * this.numStates];
        int index = 0;
        for (Node child : attach.getChildren()) {
            if (child != top) {
                otherChildren[index] = child;
                this.substitutionProcess.getTransitionProbabilities(child, childProbabilities[index]);
                index++;
            }
        }

        int[] attachStates = new int[sites.length];
        for (int k = 0; k < sites.length; k++) {
            int site = sites[k];
            int upperRow = attach.isRoot() ? 0 : this.nodeStates.getState(attach.getParent(), site);

            int[] childStates = new int[numOtherChildren];
            for (int c = 0; c < numOtherChildren; c++) {
                childStates[c] = this.nodeStates.getState(otherChildren[c], site);
            }

            double total = 0.0;
            for (int state = 0; state < this.numStates; state++) {
                double weight = upperProbabilities[upperRow * this.numStates + state];
                for (int c = 0; c < numOtherChildren; c++) {
                    weight *= childProbabilities[c][state * this.numStates + childStates[c]];
                }
                this.siteWeights[state] = weight;
                total += weight;
            }
            attachStates[k] = Categorical.drawState(this.siteWeights, 0, this.numStates, total);
        }
        return attachStates;
    }

    /* Clade */

    /** Collects the given node and all nodes below it in preorder, so that every parent comes before its children. */
    private List<Node> collectClade(Node top) {
        List<Node> clade = new ArrayList<>();
        List<Node> stack = new ArrayList<>();
        stack.add(top);
        while (!stack.isEmpty()) {
            Node node = stack.remove(stack.size() - 1);
            clade.add(node);
            stack.addAll(node.getChildren());
        }
        return clade;
    }

    /**
     * Computes the weights of leaving each state on the branch above the given node, which are
     * the off-diagonal transition probabilities, the probability ε_a to leave each state a, and
     * the bound r on these probabilities. The weights are summed from the off-diagonal entries,
     * which stays accurate on short branches.
     */
    private void prepareBranch(Node node) {
        this.substitutionProcess.getTransitionProbabilities(node, this.transitionProbabilities);

        this.maxLeaveProbability = 0.0;
        for (int from = 0; from < this.numStates; from++) {
            double leaveTotal = 0.0;
            for (int to = 0; to < this.numStates; to++) {
                double weight = from == to ? 0.0 : this.transitionProbabilities[from * this.numStates + to];
                this.leaveWeights[from * this.numStates + to] = weight;
                leaveTotal += weight;
            }
            this.leaveTotals[from] = leaveTotal;
            this.maxLeaveProbability = Math.max(this.maxLeaveProbability, leaveTotal);
        }

        this.logSkipProbability = Math.log1p(-this.maxLeaveProbability);
    }

    /**
     * Writes into the given states the states of the node of the prepared branch, which start
     * as the states of the parent, by visiting the sites at which the node may leave the state
     * of its parent.
     */
    private void drawChangedStates(int[] parentStates, int[] states) {
        for (int k = this.findNextVisit(-1, states.length); k < states.length; k = this.findNextVisit(k, states.length)) {
            int parentState = parentStates[k];
            if (Randomizer.nextDouble() * this.maxLeaveProbability < this.leaveTotals[parentState]) {
                states[k] = Categorical.drawState(this.leaveWeights, parentState * this.numStates, this.numStates, this.leaveTotals[parentState]);
            }
        }
    }

    /**
     * Returns the next position after the given one that the thinning visits, where each
     * position is visited independently with the bound on the probabilities to leave, or the given
     * end if there is none.
     */
    private int findNextVisit(int position, int end) {
        if (this.maxLeaveProbability <= 0.0) {
            return end;
        }
        if (this.maxLeaveProbability >= 1.0) {
            return Math.min(end, position + 1);
        }

        double numSkipped = Math.floor(Math.log(1.0 - Randomizer.nextDouble()) / this.logSkipProbability);
        if (numSkipped >= end - position - 1) {
            return end;
        }
        return position + 1 + (int) numSkipped;
    }

    /**
     * Returns the changes of the given node with the entries at the given sites replaced by the
     * differences between its new states and those of its parent, and the old entries at all
     * other sites kept.
     */
    private BranchChanges mergeChanges(Node node, int[] sites, int[] parentStates, int[] states) {
        BranchChanges oldChanges = this.nodeStates.getChanges(node);

        // the old entries outside the sites, and the new entries at them, both in ascending order of sites

        int[] keptSites = new int[oldChanges.size()];
        int[] keptIndices = new int[oldChanges.size()];
        int numKept = 0;
        int k = 0;
        for (int i = 0; i < oldChanges.size(); i++) {
            int site = oldChanges.getSite(i);
            while (k < sites.length && sites[k] < site) {
                k++;
            }
            if (k == sites.length || sites[k] != site) {
                keptSites[numKept] = site;
                keptIndices[numKept] = i;
                numKept++;
            }
        }

        int[] newPositions = new int[sites.length];
        int numNew = 0;
        for (int position = 0; position < sites.length; position++) {
            if (states[position] != parentStates[position]) {
                newPositions[numNew++] = position;
            }
        }

        int capacity = numKept + numNew;
        int[] mergedSites = new int[capacity];
        int[] mergedFromStates = new int[capacity];
        int[] mergedToStates = new int[capacity];
        int size = 0;
        int i = 0;
        int j = 0;
        while (i < numKept || j < numNew) {
            if (j == numNew || (i < numKept && keptSites[i] < sites[newPositions[j]])) {
                mergedSites[size] = keptSites[i];
                mergedFromStates[size] = oldChanges.getFrom(keptIndices[i]);
                mergedToStates[size] = oldChanges.getTo(keptIndices[i]);
                i++;
            } else {
                mergedSites[size] = sites[newPositions[j]];
                mergedFromStates[size] = parentStates[newPositions[j]];
                mergedToStates[size] = states[newPositions[j]];
                j++;
            }
            size++;
        }

        return BranchChanges.ofSorted(node.getNr(), mergedSites, mergedFromStates, mergedToStates, size);
    }

}
