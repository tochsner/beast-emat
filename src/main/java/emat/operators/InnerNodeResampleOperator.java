package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import emat.helper.NodeStateLookup;
import emat.helper.SiteStates;
import emat.helper.StarHistories;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.JukesCantorStochasticMapping;

import java.util.BitSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Description("Moves the height of a random non-root internal node X uniformly between its older child and its " +
        "parent, and resamples the histories on the three branches around X by approximate Jukes-Cantor stochastic " +
        "mapping (docs/mcmc-moves.md §3.5). The sequences at the parent and at both children stay fixed, so unlike " +
        "InnerNodeDisplacementOperator, the mutations on these branches do not confine the new height.")
public class InnerNodeResampleOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);

    // the columns of the star states: the outer ends P, L and R of the star around X, and its centre X
    static final int TOP = StarHistories.TOP;
    static final int LEFT = StarHistories.FIRST;
    static final int RIGHT = StarHistories.SECOND;
    static final int CENTRE = StarHistories.CENTRE;

    // the sequences at all outer ends stay fixed, so no site is free
    static final int[] NO_FREE_SITES = new int[0];
    static final BitSet NO_FREE_SITE_SET = new BitSet();
    static final int NO_FREE_COLUMN = -1;

    Mutations mutations;
    GeneticPrior geneticPrior;
    Tree tree;
    JukesCantorStochasticMapping stochasticMapping;
    StarHistories starHistories;

    // reusable states of the star around X before and after the move at the sites where not all of their states agree
    SiteStates oldStarStates;
    SiteStates newStarStates;

    // reusable lookup of the states at P for the sites whose outer end states agree
    NodeStateLookup parentStates;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.geneticPrior = this.geneticPriorInput.get();
        this.tree = this.treeInput.get();

        int numSites = this.mutations.getReferenceSequence().length;
        this.stochasticMapping = new JukesCantorStochasticMapping(this.mutations.getAlignment().getMaxStateCount(), numSites);

        this.starHistories = new StarHistories(this.stochasticMapping, numSites);
        this.oldStarStates = new SiteStates(numSites, StarHistories.NUM_COLUMNS);
        this.newStarStates = new SiteStates(numSites, StarHistories.NUM_COLUMNS);

        this.parentStates = new NodeStateLookup(this.mutations, numSites);
    }

    /**
     * Picks a non-root internal node X uniformly and draws its new height uniformly between
     * its older child and its parent P. This range does not depend on the height of X, so
     * the height proposal is symmetric. The histories on the branches above X
     * and above its children L and R are then resampled for the new height, and the
     * Hastings ratio only accounts for the proposal densities of the old and new histories.
     */
    @Override
    public double proposal() {
        int numLeaves = this.tree.getLeafNodeCount();
        int numInternalNodes = this.tree.getInternalNodeCount();
        if (numInternalNodes < 2) {
            return Double.NEGATIVE_INFINITY;
        }

        // BEAST numbers the internal nodes after the leaves, and redrawing the root keeps the choice uniform among the others
        Node x;
        do {
            x = this.tree.getNode(numLeaves + Randomizer.nextInt(numInternalNodes));
        } while (x.isRoot());

        double oldHeight = x.getHeight();
        double minHeight = Math.max(x.getLeft().getHeight(), x.getRight().getHeight());
        double newHeight = minHeight + Randomizer.nextDouble() * (x.getParent().getHeight() - minHeight);
        if (newHeight <= minHeight) {
            return Double.NEGATIVE_INFINITY;
        }

        // resample the histories while the tree still holds the old height, then move X and apply them

        Map<Node, List<Mutation>> newBranchMutations = new LinkedHashMap<>();
        double logHastingsRatio = this.proposeStarHistories(x, oldHeight, newHeight, newBranchMutations);

        x.setHeight(newHeight);

        for (Map.Entry<Node, List<Mutation>> entry : newBranchMutations.entrySet()) {
            this.mutations.applyMutations(entry.getKey(), entry.getValue(), this);
        }

        return logHastingsRatio;
    }

    /**
     * Proposes the histories on the star P–X, X–L and X–R for the new
     * height of X. The sequences at the outer ends P, L and R stay fixed. The new histories
     * are proposed under Jukes-Cantor: first the states at X given the outer ends, then the
     * history on every branch given its end states. The reverse move proposes the old
     * histories in the same way, and the genetic prior corrects for the approximate model.
     * The new mutations are put into the given map. Returns the log of
     * α_mut(n → o) / α_mut(o → n).
     */
    private double proposeStarHistories(Node x, double oldHeight, double newHeight, Map<Node, List<Mutation>> newBranchMutations) {
        Node parent = x.getParent();
        Node left = x.getLeft();
        Node right = x.getRight();

        // the fictitious Jukes-Cantor rate of a branch is its branch rate times λ(R) / L, which the move keeps

        double ratePerSite = this.geneticPrior.getTotalMutationRate(this.tree.getRoot()) / this.mutations.getReferenceSequence().length;
        double subtreeRate = this.geneticPrior.branchRateModel.getRateForBranch(x) * ratePerSite;
        double leftRate = this.geneticPrior.branchRateModel.getRateForBranch(left) * ratePerSite;
        double rightRate = this.geneticPrior.branchRateModel.getRateForBranch(right) * ratePerSite;

        List<Mutation> oldSubtreeMutations = this.mutations.getMutations(x);
        List<Mutation> oldLeftMutations = this.mutations.getMutations(left);
        List<Mutation> oldRightMutations = this.mutations.getMutations(right);

        // find the states at the outer ends, which both stars share, and at the centre X of the old star

        this.starHistories.collectOuterStates(this.mutations::getMutations, parent, left, right, this.oldStarStates);
        this.starHistories.collectCentreStates(oldSubtreeMutations, this.oldStarStates);
        this.starHistories.collectOuterStates(this.mutations::getMutations, parent, left, right, this.newStarStates);

        // sample the new histories while the tree still holds the sequences at the outer ends

        this.parentStates.resetFor(parent);

        double[] newExpectedJumps = {
                subtreeRate * (parent.getHeight() - newHeight),
                leftRate * (newHeight - left.getHeight()),
                rightRate * (newHeight - right.getHeight())
        };
        double logForwardDensity = this.stochasticMapping.sampleStarStates(
                newExpectedJumps, this.newStarStates, this.parentStates, NO_FREE_SITES, NO_FREE_SITE_SET, NO_FREE_COLUMN
        );

        List<Mutation> newSubtreeMutations = this.starHistories.sampleBranchHistory(
                x, parent.getHeight(), newHeight, subtreeRate, this.newStarStates, TOP, CENTRE, this.parentStates
        );
        List<Mutation> newLeftMutations = this.starHistories.sampleBranchHistory(
                left, newHeight, left.getHeight(), leftRate, this.newStarStates, CENTRE, LEFT, this.parentStates
        );
        List<Mutation> newRightMutations = this.starHistories.sampleBranchHistory(
                right, newHeight, right.getHeight(), rightRate, this.newStarStates, CENTRE, RIGHT, this.parentStates
        );

        logForwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                newSubtreeMutations, parent.getHeight(), newHeight, subtreeRate, this.newStarStates, TOP, CENTRE
        );
        logForwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                newLeftMutations, newHeight, left.getHeight(), leftRate, this.newStarStates, CENTRE, LEFT
        );
        logForwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                newRightMutations, newHeight, right.getHeight(), rightRate, this.newStarStates, CENTRE, RIGHT
        );

        // the old star around X, which the reverse move resamples

        double[] oldExpectedJumps = {
                subtreeRate * (parent.getHeight() - oldHeight),
                leftRate * (oldHeight - left.getHeight()),
                rightRate * (oldHeight - right.getHeight())
        };
        double logBackwardDensity = this.stochasticMapping.computeLogStarStatesDensity(
                oldExpectedJumps, this.oldStarStates, NO_FREE_SITES.length, NO_FREE_SITE_SET, NO_FREE_COLUMN
        );

        logBackwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                oldSubtreeMutations, parent.getHeight(), oldHeight, subtreeRate, this.oldStarStates, TOP, CENTRE
        );
        logBackwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                oldLeftMutations, oldHeight, left.getHeight(), leftRate, this.oldStarStates, CENTRE, LEFT
        );
        logBackwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                oldRightMutations, oldHeight, right.getHeight(), rightRate, this.oldStarStates, CENTRE, RIGHT
        );

        newBranchMutations.put(x, newSubtreeMutations);
        newBranchMutations.put(left, newLeftMutations);
        newBranchMutations.put(right, newRightMutations);

        return logBackwardDensity - logForwardDensity;
    }


}
