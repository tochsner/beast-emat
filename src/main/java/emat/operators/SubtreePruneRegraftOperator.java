package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import emat.helper.BranchMutations;
import emat.stochasticmapping.JukesCantorStochasticMapping;
import emat.state.Mutation;
import emat.helper.MutationPaths;
import emat.helper.NodeStateLookup;
import emat.helper.SiteChanges;
import emat.helper.SiteStates;
import emat.state.Mutations;
import emat.prior.GeneticPrior;

import java.util.Arrays;
import java.util.BitSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntUnaryOperator;

@Description("Prunes a subtree and regrafts it elsewhere without changing the root (docs/mcmc-moves.md §3). " +
        "By default, only the history on the branch above the pruned subtree is resampled, by approximate " +
        "Jukes-Cantor stochastic mapping (docs/mcmc-moves.md §3.4). Optionally, the histories on all branches " +
        "around the old and the new attachment point are resampled (docs/mcmc-moves.md §3.5). If the pruned " +
        "subtree is a tip, its states at the sites where its data is missing are resampled as well, so that " +
        "they follow its placement. Missations are not supported. Subclasses choose the pruned subtree and the " +
        "grafting point.")
public abstract class SubtreePruneRegraftOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<Boolean> resampleNeighbourhoodInput = new Input<>("resampleNeighbourhood", "whether to resample the histories on all branches around the old and the new attachment point instead of only on the branch above the pruned subtree", false);

    // the columns of the star states: the outer ends G, S and X of a star around P, and its centre P
    static final int TOP = 0;
    static final int SIBLING = 1;
    static final int SUBTREE = 2;
    static final int CENTRE = 3;

    Mutations mutations;
    GeneticPrior geneticPrior;
    Tree tree;
    JukesCantorStochasticMapping stochasticMapping;
    BranchMutations branchMutations;
    boolean resampleNeighbourhood;

    // reusable site maps for the changes along the paths of the current proposal and the resulting differing sites
    SiteChanges subtreeChanges;
    SiteChanges newParentChanges;
    SiteChanges differingSites;
    SiteChanges combinedChanges;

    // reusable site maps for the changes from the MRCA of the outer ends of a star down to each of them
    SiteChanges[] outerChanges;

    // reusable states of the stars around the old and the new attachment point at the sites where not all of their states agree
    SiteStates oldStarStates;
    SiteStates newStarStates;

    // reusable lookup of the states at X for the sites whose end states agree
    NodeStateLookup subtreeStates;

    // per node: the sorted sites where the data of a tip is missing, which are empty for inner nodes, and the same sites as a set
    int[][] missingSitesOfNode;
    BitSet[] missingSiteSetOfNode;

    // the missing sites of X in the current proposal, whose states at X are resampled
    int[] subtreeMissingSites;
    BitSet subtreeMissingSiteSet;

    // the model of the current proposal on the branch above X: the total mutation rate λ(R) of the root sequence times the branch rate, and the fictitious Jukes-Cantor rate μ̃
    double subtreeTotalRate;
    double jukesCantorRate;

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

        int numSites = this.mutations.getReferenceSequence().length;
        this.stochasticMapping = new JukesCantorStochasticMapping(this.mutations.getAlignment().getMaxStateCount(), numSites);
        this.branchMutations = new BranchMutations(numSites);
        this.resampleNeighbourhood = this.resampleNeighbourhoodInput.get();

        this.subtreeChanges = new SiteChanges(numSites);
        this.newParentChanges = new SiteChanges(numSites);
        this.differingSites = new SiteChanges(numSites);
        this.combinedChanges = new SiteChanges(numSites);
        this.outerChanges = new SiteChanges[]{new SiteChanges(numSites), new SiteChanges(numSites), new SiteChanges(numSites)};
        this.oldStarStates = new SiteStates(numSites, CENTRE + 1);
        this.newStarStates = new SiteStates(numSites, CENTRE + 1);
        this.subtreeStates = new NodeStateLookup(this.mutations, numSites);

        this.collectMissingSites();
    }

    /**
     * Collects the sites where the data of every tip is missing, i.e. compatible with every
     * state. Partially ambiguous characters are not included, so their states stay fixed.
     */
    private void collectMissingSites() {
        Alignment alignment = this.mutations.getAlignment();
        DataType dataType = alignment.getDataType();
        int numStates = alignment.getMaxStateCount();
        int numSites = this.mutations.getReferenceSequence().length;
        int numNodes = this.tree.getNodeCount();

        this.missingSitesOfNode = new int[numNodes][];
        this.missingSiteSetOfNode = new BitSet[numNodes];
        Arrays.fill(this.missingSitesOfNode, new int[0]);
        Arrays.fill(this.missingSiteSetOfNode, new BitSet());

        // the tips keep their numbers and taxa when the tree changes
        for (Node tip : this.tree.getExternalNodes()) {
            int taxonIndex = alignment.getTaxonIndex(tip.getID());

            BitSet missingSites = new BitSet(numSites);
            for (int site = 0; site < numSites; site++) {
                int code = alignment.getPattern(taxonIndex, alignment.getPatternIndex(site));
                if (dataType.getStatesForCode(code).length == numStates) {
                    missingSites.set(site);
                }
            }

            this.missingSitesOfNode[tip.getNr()] = missingSites.stream().toArray();
            this.missingSiteSetOfNode[tip.getNr()] = missingSites;
        }
    }

    /**
     * Detaches the subtree below X from its parent P and regrafts it with P on the branch
     * above S' at height t_P'. Neither P nor P' is the root. If S' is the old sibling S, the
     * topology stays the same and only the height of P shifts along the joined G–S branch.
     */
    @Override
    public double proposal() {
        Node x = this.pickSubtreeRoot();
        if (x == null) {
            return Double.NEGATIVE_INFINITY;
        }

        this.updateModel(x);
        this.subtreeMissingSites = this.missingSitesOfNode[x.getNr()];
        this.subtreeMissingSiteSet = this.missingSiteSetOfNode[x.getNr()];

        // choose a grafting point

        GraftingPoint graftingPoint = this.proposeGraftingPoint(x);
        if (graftingPoint == null || !this.isValidGraftingPoint(x, graftingPoint)) {
            return Double.NEGATIVE_INFINITY;
        }

        // regraft and resample the local history

        double logMutationHastingsRatio = this.resampleNeighbourhood
                ? this.regraftWithNeighbourhoodHistories(x, graftingPoint)
                : this.regraftWithSubtreeHistory(x, graftingPoint);

        return graftingPoint.logHastingsRatio() + logMutationHastingsRatio;
    }

    /* Grafting Point Proposals */

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

    /* Grafting and Resampling */

    /**
     * Regrafts the subtree and resamples only the history on the new P'–X branch. The
     * mutations of the old P branch are merged into the branch of the old sibling S, the
     * branch above S' is split at t_P', and everything else is kept. At the missing sites of
     * X, the states at X are first sampled from the states at P' under Jukes-Cantor, and the
     * history is then sampled given the end states. Returns the log of
     * α_mut(n → o) / α_mut(o → n).
     */
    private double regraftWithSubtreeHistory(Node x, GraftingPoint graftingPoint) {
        Node parent = x.getParent();
        Node sibling = this.getOtherChild(parent, x);
        Node grandparent = parent.getParent();

        Node newSibling = graftingPoint.newSibling();
        boolean isHeightShift = newSibling == sibling;
        double newParentHeight = graftingPoint.newParentHeight();
        double oldParentHeight = parent.getHeight();

        // find the sites that differ between the ends of the old P–X branch, which are exactly the changes on it

        List<Mutation> oldSubtreeMutations = this.mutations.getMutations(x);
        MutationPaths.collectChanges(this.mutations, parent, x, x.getHeight(), this.subtreeChanges);
        int oldNumDifferingSites = MutationPaths.countDifferingSites(this.subtreeChanges);
        int oldNumDifferingMissingSites = this.countDifferingMissingSites(this.subtreeChanges);

        // P' lies on the branch above this node in the current tree, which is the old P branch if P shifts upwards

        Node newParentBranchNode = isHeightShift && newParentHeight > oldParentHeight ? parent : newSibling;

        // find the MRCA of X and P', then the changes from it down to both, which the new P'–X branch has to compensate

        Node mrca = MutationPaths.findMrca(x, newParentBranchNode.getParent());
        MutationPaths.collectChanges(this.mutations, mrca, x, x.getHeight(), this.subtreeChanges);
        MutationPaths.collectChanges(this.mutations, mrca, newParentBranchNode, newParentHeight, this.newParentChanges);
        MutationPaths.combineChanges(this.newParentChanges, this.subtreeChanges, this.combinedChanges);

        // keep the differing sites where the data of X is observed, and sample the states of X at its missing sites

        this.differingSites.clear();
        for (int slot = 0; slot < this.combinedChanges.getSize(); slot++) {
            int site = this.combinedChanges.getSite(slot);
            if (!this.subtreeMissingSiteSet.get(site)) {
                this.differingSites.addSite(site, this.combinedChanges.getStartState(slot), this.combinedChanges.getEndState(slot));
            }
        }

        this.subtreeStates.resetFor(x);
        IntUnaryOperator newParentStates = this::getNewParentState;

        double newExpectedJumps = this.jukesCantorRate * (newParentHeight - x.getHeight());
        int newNumDifferingMissingSites = this.stochasticMapping.sampleFreeEndStates(
                newExpectedJumps, this.subtreeMissingSites, newParentStates, this.differingSites
        );

        // sample the history on the new P'–X branch while the tree still holds the old sequences, where every agreeing site is in the state of P'

        List<Mutation> newSubtreeMutations = this.stochasticMapping.sampleBranchHistory(
                x.getNr(), newParentHeight, x.getHeight(), this.jukesCantorRate, this.differingSites, newParentStates
        );

        // rearrange the mutations: join G–P–S into G–S, then split G'–S' into G'–P'–S'

        Map<Node, List<Mutation>> newBranchMutations = new LinkedHashMap<>();

        List<Mutation> joinedMutations = this.branchMutations.joinBranches(
                this.mutations.getMutations(parent), this.mutations.getMutations(sibling), sibling.getNr(), grandparent.getHeight()
        );
        newBranchMutations.put(sibling, joinedMutations);

        List<Mutation> newSiblingMutations = isHeightShift ? joinedMutations : this.mutations.getMutations(newSibling);
        BranchMutations.Split split = this.branchMutations.splitBranch(newSiblingMutations, newParentHeight, parent.getNr(), newSibling.getNr());
        newBranchMutations.put(parent, split.upperMutations());
        newBranchMutations.put(newSibling, split.lowerMutations());

        this.regraft(x, newSibling, newParentHeight);

        for (Map.Entry<Node, List<Mutation>> entry : newBranchMutations.entrySet()) {
            this.mutations.applyMutations(entry.getKey(), entry.getValue(), this);
        }

        this.mutations.applyMutations(x, newSubtreeMutations, this);

        // calculate the hastings correction

        double oldExpectedJumps = this.jukesCantorRate * (oldParentHeight - x.getHeight());

        double logForwardDensity = this.computeLogBranchHistoryDensity(x, newParentHeight, newSubtreeMutations, this.differingSites.getSize())
                + this.stochasticMapping.computeLogFreeEndStatesDensity(newExpectedJumps, this.subtreeMissingSites.length, newNumDifferingMissingSites);
        double logBackwardDensity = this.computeLogBranchHistoryDensity(x, oldParentHeight, oldSubtreeMutations, oldNumDifferingSites)
                + this.stochasticMapping.computeLogFreeEndStatesDensity(oldExpectedJumps, this.subtreeMissingSites.length, oldNumDifferingMissingSites);

        return logBackwardDensity - logForwardDensity;
    }

    /**
     * Regrafts the subtree and resamples the histories on all branches around the old and
     * the new attachment point (docs/mcmc-moves.md §3.5). These are the branches above P,
     * S, X and S', which form the star G–P, P–S, P–X and the branch G'–S' before the move,
     * and the joined branch G–S and the star G'–P', P'–S', P'–X after it. The sequences at
     * the outer ends G, S, X, G' and S' stay fixed, except at the missing sites of X. The
     * new histories are proposed under Jukes-Cantor: first the states at P' given the outer
     * ends of its star, then the states of X at its missing sites given P', then the history
     * on every branch given its end states. The reverse move proposes the old
     * histories in the same way, and the genetic prior corrects for the approximate model.
     * Returns the log of α_mut(n → o) / α_mut(o → n).
     */
    private double regraftWithNeighbourhoodHistories(Node x, GraftingPoint graftingPoint) {
        Node parent = x.getParent();
        Node sibling = this.getOtherChild(parent, x);
        Node grandparent = parent.getParent();

        Node newSibling = graftingPoint.newSibling();
        Node newGrandparent = this.getPrunedParent(x, newSibling);
        boolean isHeightShift = newSibling == sibling;
        double newParentHeight = graftingPoint.newParentHeight();
        double oldParentHeight = parent.getHeight();

        // the fictitious Jukes-Cantor rate of a branch is its branch rate times λ(R) / L, which the move keeps as the branches keep the rates of the nodes below them

        double ratePerSite = this.geneticPrior.getTotalMutationRate(this.tree.getRoot()) / this.mutations.getReferenceSequence().length;
        double parentRate = this.geneticPrior.branchRateModel.getRateForBranch(parent) * ratePerSite;
        double siblingRate = this.geneticPrior.branchRateModel.getRateForBranch(sibling) * ratePerSite;
        double newSiblingRate = this.geneticPrior.branchRateModel.getRateForBranch(newSibling) * ratePerSite;
        double subtreeRate = this.jukesCantorRate;

        List<Mutation> oldParentMutations = this.mutations.getMutations(parent);
        List<Mutation> oldSiblingMutations = this.mutations.getMutations(sibling);
        List<Mutation> oldSubtreeMutations = this.mutations.getMutations(x);
        List<Mutation> oldNewSiblingMutations = this.mutations.getMutations(newSibling);

        // find the states at the outer ends of both stars, and at the centre P of the old star

        this.collectOuterStates(grandparent, sibling, x, this.oldStarStates);
        this.collectCentreStates(oldParentMutations, this.oldStarStates);
        this.collectOuterStates(newGrandparent, newSibling, x, this.newStarStates);

        // sample the new histories while the tree still holds the sequences at the outer ends, which the move keeps

        this.subtreeStates.resetFor(x);

        double[] newExpectedJumps = {
                parentRate * (newGrandparent.getHeight() - newParentHeight),
                newSiblingRate * (newParentHeight - newSibling.getHeight()),
                subtreeRate * (newParentHeight - x.getHeight())
        };
        double logForwardDensity = this.stochasticMapping.sampleStarStates(
                newExpectedJumps, this.newStarStates, this.subtreeStates, this.subtreeMissingSites, this.subtreeMissingSiteSet, SUBTREE
        );

        List<Mutation> newParentMutations = this.sampleStarBranchHistory(
                parent, newGrandparent.getHeight(), newParentHeight, parentRate, this.newStarStates, TOP, CENTRE
        );
        List<Mutation> newSiblingMutations = this.sampleStarBranchHistory(
                newSibling, newParentHeight, newSibling.getHeight(), newSiblingRate, this.newStarStates, CENTRE, SIBLING
        );
        List<Mutation> newSubtreeMutations = this.sampleStarBranchHistory(
                x, newParentHeight, x.getHeight(), subtreeRate, this.newStarStates, CENTRE, SUBTREE
        );

        logForwardDensity += this.computeLogStarBranchDensity(
                newParentMutations, newGrandparent.getHeight(), newParentHeight, parentRate, this.newStarStates, TOP, CENTRE
        );
        logForwardDensity += this.computeLogStarBranchDensity(
                newSiblingMutations, newParentHeight, newSibling.getHeight(), newSiblingRate, this.newStarStates, CENTRE, SIBLING
        );
        logForwardDensity += this.computeLogStarBranchDensity(
                newSubtreeMutations, newParentHeight, x.getHeight(), subtreeRate, this.newStarStates, CENTRE, SUBTREE
        );

        // the old star around P, which the reverse move resamples

        double[] oldExpectedJumps = {
                parentRate * (grandparent.getHeight() - oldParentHeight),
                siblingRate * (oldParentHeight - sibling.getHeight()),
                subtreeRate * (oldParentHeight - x.getHeight())
        };
        double logBackwardDensity = this.stochasticMapping.computeLogStarStatesDensity(
                oldExpectedJumps, this.oldStarStates, this.subtreeMissingSites.length, this.subtreeMissingSiteSet, SUBTREE
        );

        logBackwardDensity += this.computeLogStarBranchDensity(
                oldParentMutations, grandparent.getHeight(), oldParentHeight, parentRate, this.oldStarStates, TOP, CENTRE
        );
        logBackwardDensity += this.computeLogStarBranchDensity(
                oldSiblingMutations, oldParentHeight, sibling.getHeight(), siblingRate, this.oldStarStates, CENTRE, SIBLING
        );
        logBackwardDensity += this.computeLogStarBranchDensity(
                oldSubtreeMutations, oldParentHeight, x.getHeight(), subtreeRate, this.oldStarStates, CENTRE, SUBTREE
        );

        // unless P only shifts, the move also joins G–S, and the reverse move joins G'–S'

        List<Mutation> joinedMutations = null;
        if (!isHeightShift) {
            joinedMutations = this.sampleStarBranchHistory(
                    sibling, grandparent.getHeight(), sibling.getHeight(), siblingRate, this.oldStarStates, TOP, SIBLING
            );
            logForwardDensity += this.computeLogStarBranchDensity(
                    joinedMutations, grandparent.getHeight(), sibling.getHeight(), siblingRate, this.oldStarStates, TOP, SIBLING
            );
            logBackwardDensity += this.computeLogStarBranchDensity(
                    oldNewSiblingMutations, newGrandparent.getHeight(), newSibling.getHeight(), newSiblingRate, this.newStarStates, TOP, SIBLING
            );
        }

        this.regraft(x, newSibling, newParentHeight);

        this.mutations.applyMutations(parent, newParentMutations, this);
        this.mutations.applyMutations(newSibling, newSiblingMutations, this);
        this.mutations.applyMutations(x, newSubtreeMutations, this);
        if (!isHeightShift) {
            this.mutations.applyMutations(sibling, joinedMutations, this);
        }

        return logBackwardDensity - logForwardDensity;
    }

    /* Stochastic Mapping */

    /**
     * Takes the current model from the genetic prior. The fictitious Jukes-Cantor rate is
     * μ̃ = λ(R) / L, with λ(R) the total mutation rate of the root sequence times the branch
     * rate of X, which the genetic prior has cached for the current state. The move keeps
     * the root sequence and the branch rate of X, so the reverse move uses the same model,
     * even though the sequence of X changes at its missing sites.
     */
    private void updateModel(Node x) {
        double branchRate = this.geneticPrior.branchRateModel.getRateForBranch(x);
        int numSites = this.mutations.getReferenceSequence().length;

        this.subtreeTotalRate = branchRate * this.geneticPrior.getTotalMutationRate(this.tree.getRoot());
        this.jukesCantorRate = this.subtreeTotalRate / numSites;
    }

    /**
     * Returns the state at P' of the current proposal. The changes from the MRCA of X and P'
     * down to both must be collected, and the lookup of the states at X be reset.
     */
    private int getNewParentState(int site) {
        int slot = this.newParentChanges.getSlot(site);
        if (slot >= 0) {
            return this.newParentChanges.getEndState(slot);
        }

        // without changes towards P', P' keeps the state at the MRCA, which is also the state of X unless it changes towards X
        slot = this.subtreeChanges.getSlot(site);
        return slot >= 0 ? this.subtreeChanges.getStartState(slot) : this.subtreeStates.applyAsInt(site);
    }

    /** Counts the missing sites of X whose states differ between the start and the end of the given changes. */
    private int countDifferingMissingSites(SiteChanges changes) {
        int numDifferingMissingSites = 0;
        for (int slot = 0; slot < changes.getSize(); slot++) {
            if (changes.getStartState(slot) != changes.getEndState(slot) && this.subtreeMissingSiteSet.get(changes.getSite(slot))) {
                numDifferingMissingSites++;
            }
        }
        return numDifferingMissingSites;
    }

    /**
     * Computes the log probability α_mut that the Jukes-Cantor mapping with the rate μ̃
     * proposes the given history on the branch above X that starts at the given height,
     * given the number of sites whose states differ between its ends.
     */
    protected double computeLogBranchHistoryDensity(Node x, double startHeight, List<Mutation> branchMutations, int numDifferingSites) {
        return this.stochasticMapping.computeLogBranchHistoryDensity(
                branchMutations, startHeight, x.getHeight(), this.jukesCantorRate, numDifferingSites
        );
    }

    /* Neighbourhood Histories */

    /**
     * Collects the states at the outer ends G, S and X of a star at every site where they
     * do not all agree, from the changes on the paths from their MRCA down to each of them.
     * The centre states are left unset.
     */
    private void collectOuterStates(Node top, Node sibling, Node x, SiteStates states) {
        Node[] ends = {top, sibling, x};
        Node mrca = MutationPaths.findMrca(MutationPaths.findMrca(x, sibling), top);

        for (int i = 0; i < ends.length; i++) {
            MutationPaths.collectChanges(this.mutations, mrca, ends[i], ends[i].getHeight(), this.outerChanges[i]);
        }

        states.clear();
        int[] endStates = new int[ends.length];

        for (SiteChanges changes : this.outerChanges) {
            for (int slot = 0; slot < changes.getSize(); slot++) {
                int site = changes.getSite(slot);
                if (states.containsSite(site)) {
                    continue;
                }

                // an end without changes at the site keeps the state at the MRCA
                int mrcaState = changes.getStartState(slot);
                boolean isAgreeing = true;
                for (int i = 0; i < ends.length; i++) {
                    int endSlot = this.outerChanges[i].getSlot(site);
                    endStates[i] = endSlot >= 0 ? this.outerChanges[i].getEndState(endSlot) : mrcaState;
                    isAgreeing &= endStates[i] == endStates[0];
                }

                if (!isAgreeing) {
                    int statesSlot = states.addSite(site);
                    for (int i = 0; i < ends.length; i++) {
                        states.setState(statesSlot, i, endStates[i]);
                    }
                }
            }
        }
    }

    /**
     * Completes the states of a star with the states at its centre, which follow from the
     * states at the top end and the mutations on the branch from it down to the centre.
     * Sites where only the centre differs from the agreeing outer ends are added.
     */
    private void collectCentreStates(List<Mutation> topMutations, SiteStates states) {
        for (int slot = 0; slot < states.getSize(); slot++) {
            states.setState(slot, CENTRE, states.getState(slot, TOP));
        }

        // the mutations are sorted by descending height, so the first one at a site starts in the top state and the last one sets the centre state
        for (Mutation mutation : topMutations) {
            int slot = states.getSlot(mutation.site());
            if (slot < 0) {
                slot = states.addSite(mutation.site());
                states.setAllStates(slot, mutation.oldState());
            }
            states.setState(slot, CENTRE, mutation.newState());
        }
    }

    /**
     * Samples a new history on the branch above the given node, whose end states are held
     * by the given columns of the star states. At the sites without an entry, both ends are
     * in the state of X. The returned mutations are sorted by descending height.
     */
    private List<Mutation> sampleStarBranchHistory(Node node, double startHeight, double endHeight, double mutationRate,
                                                   SiteStates states, int startColumn, int endColumn) {
        this.differingSites.clear();
        for (int slot = 0; slot < states.getSize(); slot++) {
            int startState = states.getState(slot, startColumn);
            int endState = states.getState(slot, endColumn);
            if (startState != endState) {
                this.differingSites.addSite(states.getSite(slot), startState, endState);
            }
        }

        return this.stochasticMapping.sampleBranchHistory(
                node.getNr(), startHeight, endHeight, mutationRate, this.differingSites,
                site -> {
                    int slot = states.getSlot(site);
                    return slot >= 0 ? states.getState(slot, startColumn) : this.subtreeStates.applyAsInt(site);
                }
        );
    }

    /**
     * Computes the log probability that sampleStarBranchHistory proposes the given history
     * on a branch whose end states are held by the given columns of the star states.
     */
    private double computeLogStarBranchDensity(List<Mutation> branchMutations, double startHeight, double endHeight,
                                               double mutationRate, SiteStates states, int startColumn, int endColumn) {
        int numDifferingSites = 0;
        for (int slot = 0; slot < states.getSize(); slot++) {
            if (states.getState(slot, startColumn) != states.getState(slot, endColumn)) {
                numDifferingSites++;
            }
        }

        return this.stochasticMapping.computeLogBranchHistoryDensity(
                branchMutations, startHeight, endHeight, mutationRate, numDifferingSites
        );
    }

    /* Tree Rearrangement */

    /**
     * Moves P with the subtree below X onto the branch above S' at the given height. If S'
     * is the old sibling, only the height of P changes.
     */
    private void regraft(Node x, Node newSibling, double newParentHeight) {
        Node parent = x.getParent();
        Node sibling = this.getOtherChild(parent, x);
        Node grandparent = parent.getParent();
        Node newGrandparent = this.getPrunedParent(x, newSibling);

        if (newSibling != sibling) {
            this.replace(grandparent, parent, sibling);
            this.replace(parent, sibling, newSibling);
            this.replace(newGrandparent, newSibling, parent);
        }
        parent.setHeight(newParentHeight);
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
