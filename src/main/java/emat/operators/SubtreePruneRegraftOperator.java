package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import emat.helper.BranchMutations;
import emat.state.Mutation;
import emat.helper.MutationPaths;
import emat.state.Mutations;
import emat.helper.StochasticMapping;
import emat.prior.GeneticPrior;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Description("Prunes a subtree and regrafts it elsewhere without changing the root (docs/mcmc-moves.md §3). " +
        "Only the history on the branch above the pruned subtree is resampled, by exact stochastic mapping " +
        "under the model of the genetic prior. Missations are not supported. " +
        "Subclasses choose the pruned subtree and the grafting point.")
public abstract class SubtreePruneRegraftOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);

    Mutations mutations;
    GeneticPrior geneticPrior;
    Tree tree;
    StochasticMapping stochasticMapping;

    // the model of the current proposal on the branch above X: R = rate scale * Q, and the eigen decomposition of Q
    double rateScale;
    double[] branchRateMatrix;
    EigenDecomposition eigenDecomposition;

    /**
     * A new attachment point for the pruned subtree: the branch above the new sibling S' in
     * the pruned tree, the height of the new parent P' on it, and the log of
     * α_graft(n → o) / α_graft(o → n).
     */
    protected record GraftingPoint(Node newSibling, double newParentHeight, double logHastingsRatio) {
    }

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.geneticPrior = this.geneticPriorInput.get();
        this.tree = this.treeInput.get();
        this.stochasticMapping = new StochasticMapping(this.mutations.getAlignment().getMaxStateCount());
    }

    /**
     * Detaches the subtree below X from its parent P and regrafts it with P on the branch
     * above S' at height t_P'. Neither P nor P' is the root. The mutations of the old P
     * branch are merged into the branch of the old sibling S, the branch above S' is split
     * at t_P', and the history on the new P'–X branch is resampled. Everything else is kept.
     * If S' is the old sibling S, the topology stays the same and only the height of P
     * shifts along the joined G–S branch.
     */
    @Override
    public double proposal() {
        Node x = this.pickSubtreeRoot();
        if (x == null) {
            return Double.NEGATIVE_INFINITY;
        }

        Node parent = x.getParent();
        Node sibling = this.getOtherChild(parent, x);
        Node grandparent = parent.getParent();

        GraftingPoint graftingPoint = this.proposeGraftingPoint(x);
        if (graftingPoint == null || !this.isValidGraftingPoint(x, graftingPoint)) {
            return Double.NEGATIVE_INFINITY;
        }

        Node newSibling = graftingPoint.newSibling();
        Node newGrandparent = this.getPrunedParent(x, newSibling);
        boolean isHeightShift = newSibling == sibling;
        double newParentHeight = graftingPoint.newParentHeight();
        double oldParentHeight = parent.getHeight();

        this.updateModel(x);

        // find the sites that differ between the ends of the old P–X branch, which are exactly the changes on it

        List<Mutation> oldSubtreeMutations = this.mutations.getMutations(x);
        Map<Integer, int[]> oldDifferingSites = MutationPaths.combineChanges(
                new HashMap<>(), MutationPaths.collectChanges(this.mutations, parent, x, x.getHeight())
        );

        // P' lies on the branch above this node in the current tree, which is the old P branch if P shifts upwards

        Node newParentBranchNode = isHeightShift && newParentHeight > oldParentHeight ? parent : newSibling;

        // find the MRCA of X and P', then the changes from it down to both, which the new P'–X branch has to compensate

        Node mrca = MutationPaths.findMrca(x, newParentBranchNode.getParent());
        Map<Integer, int[]> subtreeChanges = MutationPaths.collectChanges(this.mutations, mrca, x, x.getHeight());
        Map<Integer, int[]> newParentChanges = MutationPaths.collectChanges(this.mutations, mrca, newParentBranchNode, newParentHeight);
        Map<Integer, int[]> newDifferingSites = MutationPaths.combineChanges(newParentChanges, subtreeChanges);

        // the exact mapping needs the full sequences at both ends of the old and the new P–X branch

        int[] subtreeSequence = MutationPaths.getSequence(this.mutations, x);
        int[] oldParentSequence = MutationPaths.getStartSequence(subtreeSequence, oldDifferingSites);
        int[] newParentSequence = MutationPaths.getStartSequence(subtreeSequence, newDifferingSites);

        // rearrange the mutations: join G–P–S into G–S, then split G'–S' into G'–P'–S'

        Map<Node, List<Mutation>> newBranchMutations = new LinkedHashMap<>();

        List<Mutation> joinedMutations = BranchMutations.joinBranches(
                this.mutations.getMutations(parent), this.mutations.getMutations(sibling), sibling.getNr(), grandparent.getHeight()
        );
        newBranchMutations.put(sibling, joinedMutations);

        List<Mutation> newSiblingMutations = isHeightShift ? joinedMutations : this.mutations.getMutations(newSibling);
        BranchMutations.Split split = BranchMutations.splitBranch(newSiblingMutations, newParentHeight, parent.getNr(), newSibling.getNr());
        newBranchMutations.put(parent, split.upperMutations());
        newBranchMutations.put(newSibling, split.lowerMutations());

        // rearrange the tree, which only moves P for a height shift

        if (!isHeightShift) {
            this.replace(grandparent, parent, sibling);
            this.replace(parent, sibling, newSibling);
            this.replace(newGrandparent, newSibling, parent);
        }
        parent.setHeight(newParentHeight);

        for (Map.Entry<Node, List<Mutation>> entry : newBranchMutations.entrySet()) {
            this.mutations.applyMutations(entry.getKey(), entry.getValue(), this);
        }

        // resample the history on the new P'–X branch

        List<Mutation> newSubtreeMutations = this.sampleBranchHistory(x, newParentHeight, newParentSequence, subtreeSequence);
        this.mutations.applyMutations(x, newSubtreeMutations, this);

        // combine the grafting and the mutation proposal probabilities

        double logForwardDensity = this.computeLogBranchHistoryDensity(x, newParentHeight, newSubtreeMutations, newParentSequence, subtreeSequence);
        double logBackwardDensity = this.computeLogBranchHistoryDensity(x, oldParentHeight, oldSubtreeMutations, oldParentSequence, subtreeSequence);

        return graftingPoint.logHastingsRatio() + logBackwardDensity - logForwardDensity;
    }

    /* Grafting Strategy */

    /**
     * Picks the root X of the subtree to prune, such that neither X nor its parent is the
     * root. Returns null if there is no such node. The choice must be accounted for in the
     * Hastings ratio unless the set of candidates does not change under the move.
     */
    protected abstract Node pickSubtreeRoot();

    /**
     * Proposes the new attachment point of the subtree below X in the pruned tree, i.e. with
     * X and its parent P removed and P's two branches joined. Returns null if there is no
     * valid proposal. Candidates are the subtree slide, Wilson–Balding and mdSPR proposals of
     * docs/mcmc-moves.md §8.
     */
    protected abstract GraftingPoint proposeGraftingPoint(Node x);

    /* Stochastic Mapping */

    /**
     * Takes the current model from the genetic prior. The branch above X keeps its rate
     * under the move, so the rate matrix R of the old and the new P–X branch is the same.
     */
    private void updateModel(Node x) {
        Node root = this.tree.getRoot();

        this.rateScale = this.geneticPrior.branchRateModel.getRateForBranch(x)
                * this.geneticPrior.siteModel.getRateForCategory(0, root);
        this.eigenDecomposition = this.geneticPrior.substitutionModel.getEigenDecomposition(root);

        this.branchRateMatrix = this.geneticPrior.computeRateMatrix();
        for (int i = 0; i < this.branchRateMatrix.length; i++) {
            this.branchRateMatrix[i] *= this.rateScale;
        }
    }

    /**
     * Samples a new history on the branch above X that starts at the given height, given
     * the full sequences at both of its ends. This maps every site exactly under the model
     * of the genetic prior (docs/mcmc-moves.md §3.3), so the genetic prior of the branch
     * cancels in the acceptance probability, but its cost scales with the genome length.
     * The returned mutations belong to X and are sorted by descending height.
     */
    protected List<Mutation> sampleBranchHistory(Node x, double startHeight, int[] startSequence, int[] endSequence) {
        double[] transitionProbabilities = this.computeBranchTransitionProbabilities(x, startHeight);
        return this.stochasticMapping.sampleSequenceHistory(
                x.getNr(), startHeight, x.getHeight(), startSequence, endSequence, this.branchRateMatrix, transitionProbabilities
        );
    }

    /**
     * Computes the log probability α_mut that sampleBranchHistory proposes the given history
     * on the branch above X that starts at the given height, given the full sequences at
     * both of its ends.
     */
    protected double computeLogBranchHistoryDensity(Node x, double startHeight, List<Mutation> branchMutations,
                                                    int[] startSequence, int[] endSequence) {
        double[] transitionProbabilities = this.computeBranchTransitionProbabilities(x, startHeight);
        return this.stochasticMapping.computeLogSequenceHistoryDensity(
                branchMutations, startHeight, x.getHeight(), startSequence, endSequence, this.branchRateMatrix, transitionProbabilities
        );
    }

    /** Computes the transition probabilities exp(R t) of a branch from the given height down to X. */
    private double[] computeBranchTransitionProbabilities(Node x, double startHeight) {
        return this.stochasticMapping.computeTransitionProbabilities(this.eigenDecomposition, this.rateScale * (startHeight - x.getHeight()));
    }

    /* Tree Rearrangement */

    /**
     * Checks that the grafting point lies on a branch of the pruned tree that does not end
     * at the root, and that the new parent is older than X and the new sibling and younger
     * than the new grandparent.
     */
    private boolean isValidGraftingPoint(Node x, GraftingPoint graftingPoint) {
        Node parent = x.getParent();
        Node newSibling = graftingPoint.newSibling();
        double newParentHeight = graftingPoint.newParentHeight();

        if (newSibling == parent || newSibling.isRoot() || this.isInSubtree(newSibling, x)) {
            return false;
        }

        return newParentHeight > x.getHeight()
                && newParentHeight > newSibling.getHeight()
                && newParentHeight < this.getPrunedParent(x, newSibling).getHeight();
    }

    /**
     * Returns the parent of the given node in the pruned tree, where X and its parent P are
     * removed and the old sibling S hangs directly below the old grandparent G.
     */
    protected Node getPrunedParent(Node x, Node node) {
        Node parent = x.getParent();
        return node.getParent() == parent ? parent.getParent() : node.getParent();
    }

    /** Checks whether the given node lies in the subtree below the given root, including the root itself. */
    private boolean isInSubtree(Node node, Node subtreeRoot) {
        for (Node ancestor = node; ancestor != null; ancestor = ancestor.getParent()) {
            if (ancestor == subtreeRoot) {
                return true;
            }
        }
        return false;
    }

}
