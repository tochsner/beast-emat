package esat.operators.topology;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import emat.operators.topology.mdspr.BranchLengthSubtreeSelector;
import esat.helper.TipCompatibility;
import esat.prior.GeneticPrior;
import esat.state.BranchChanges;
import esat.state.NodeStates;
import esat.stochasticmapping.NodeStateSampler;
import esat.stochasticmapping.StateEdit;
import esat.stochasticmapping.SubstitutionProcess;
import org.apache.commons.statistics.distribution.BetaDistribution;

import java.util.Arrays;

@Description("Regrafts a subtree, picked proportional to the length of the branch above it, together with its parent " +
        "onto another branch, and redraws the states of the parent from their full conditional given its new neighbours. " +
        "All other node states are kept. Each branch of the pruned tree is scored as if the parent took the majority state of the " +
        "ends of the branch and of the subtree: the differences between the subtree and the ends of the branch are counted " +
        "as changes under a Jukes-Cantor model, with the rate and the distances of the evolutionary model. " +
        "The missing sites of a tip, and its ambiguous sites among the states compatible with its data, never count as differences. " +
        "Most proposals only consider the branches connected to the current attachment point through branches at which " +
        "the subtree differs from both ends at few sites. " +
        "The height of the parent on the new branch is drawn from a Beta distribution given by the changes " +
        "above and below the parent if it took that majority state. The root is never changed.")
public class StateDirectedSprOperator extends TreeOperator {

    final public Input<NodeStates> nodeStatesInput = new Input<>("nodeStates", "the node states, of which the move redraws those of the parent of the pruned subtree", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<Double> fullExplorationProbabilityInput = new Input<>("fullExplorationProbability", "the probability of considering every branch instead of the neighbourhood of the current attachment point", 0.01);
    final public Input<Double> windowProbabilityInput = new Input<>("windowProbability", "the parameter q of the size of the neighbourhood, which grows by j geometric with probability q^j (1 - q)", 0.5);
    final public Input<Double> annealingInput = new Input<>("annealing", "the power f to which the scores of the branches and the density of the height are raised, which makes the proposals less eager", 0.8);
    final public Input<Boolean> scaleWeightByLogNumTipsInput = new Input<>("scaleWeightByLogNumTips", "whether to scale the weight by the natural logarithm of the number of tips n, so that the operator is selected with weight * ln(n). Otherwise, the weight is used as given", false);

    Tree tree;
    NodeStates nodeStates;
    SubstitutionProcess substitutionProcess;
    NodeStateSampler nodeStateSampler;
    TipCompatibility tipCompatibility;
    BranchLengthSubtreeSelector subtreeSelector;

    double fullExplorationProbability;
    double windowProbability;
    double annealing;

    // the factor ln(n) or one by which the weight of the operator is scaled
    double weightFactor;

    // the subtree root X of the current proposal, its parent P that moves with it, and its sibling S
    Node x;
    Node parent;
    Node sibling;

    // per node: the proposal that scored the branch above it, its log score, the changes above and below P attached to it, and the sites at which X differs from both ends
    int proposalStamp;
    int[] scoreStamps;
    double[] logScores;
    int[] upperNumDifferences;
    int[] lowerNumDifferences;
    int[] subtreeNumDifferences;

    // per node: the exploration that reached the branch above it, and the reached branches in the order they were reached
    int explorationStamp;
    int[] explorationStamps;
    Node[] candidates;
    int numCandidates;

    // the scores of the candidates relative to the largest one, and their sum
    double[] candidateScores;
    double totalCandidateScore;

    // the states that count as equal to X at each site in the current proposal, as bit masks: the state of X, or the compatible states of a tip at its missing and ambiguous sites
    int[] xStates;
    int numSites;

    // per node, the number of sites at which it differs from X, valid if the proposal stamped it
    int[] numXDifferences;
    int[] xDifferenceStamps;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.nodeStates = this.nodeStatesInput.get();
        this.substitutionProcess = new SubstitutionProcess(this.geneticPriorInput.get());
        this.nodeStateSampler = new NodeStateSampler(this.nodeStates, this.substitutionProcess);
        this.tipCompatibility = this.geneticPriorInput.get().getTipCompatibility();
        this.subtreeSelector = new BranchLengthSubtreeSelector(this.tree);

        this.fullExplorationProbability = this.fullExplorationProbabilityInput.get();
        this.windowProbability = this.windowProbabilityInput.get();
        this.annealing = this.annealingInput.get();
        if (this.windowProbability < 0.0 || this.windowProbability >= 1.0) {
            throw new IllegalArgumentException("windowProbability must be at least 0 and less than 1.");
        }
        if (this.annealing <= 0.0) {
            throw new IllegalArgumentException("annealing must be positive.");
        }

        this.weightFactor = this.scaleWeightByLogNumTipsInput.get() ? Math.log(this.tree.getLeafNodeCount()) : 1.0;

        int numNodes = this.tree.getNodeCount();
        this.scoreStamps = new int[numNodes];
        this.logScores = new double[numNodes];
        this.upperNumDifferences = new int[numNodes];
        this.lowerNumDifferences = new int[numNodes];
        this.subtreeNumDifferences = new int[numNodes];
        this.explorationStamps = new int[numNodes];
        this.candidates = new Node[numNodes];
        this.candidateScores = new double[numNodes];
        this.numSites = this.nodeStates.getAlignment().getSiteCount();
        this.numXDifferences = new int[numNodes];
        this.xDifferenceStamps = new int[numNodes];
    }

    /** Returns the weight, scaled by the natural logarithm of the number of tips if requested. */
    @Override
    public double getWeight() {
        return super.getWeight() * this.weightFactor;
    }

    /**
     * Moves P with the subtree below X onto a branch of the pruned tree, i.e. the tree
     * without X and P in which G–P–S is joined into G–S, and redraws the states of P from
     * their full conditional given its new neighbours. The branch is drawn proportional to its
     * score among the candidates, and the height of P on it from a Beta distribution, both
     * of which only depend on the pruned tree and X, not on the states of P. The pruned tree
     * is the same before and after the move, so the reverse move draws the current branch from
     * the candidates around the new one with the same limit and the same scores, as the
     * neighbourhood is the same from all of its branches. The reverse move redraws
     * the old states of P from their full conditional given the old neighbours.
     */
    @Override
    public double proposal() {
        Node x = this.subtreeSelector.pick();
        if (x == null) {
            return Double.NEGATIVE_INFINITY;
        }
        double logForwardPickProbability = this.subtreeSelector.computeLogProbability(x);

        this.x = x;
        this.parent = x.getParent();
        this.sibling = this.getOtherChild(this.parent, x);
        this.substitutionProcess.update();
        this.startProposal();
        this.xStates = this.computeXStates();

        boolean isFullExploration = Randomizer.nextDouble() < this.fullExplorationProbability;
        int limit = isFullExploration ? Integer.MAX_VALUE : this.drawLimit();

        // draw the new branch and the new height of P on it

        this.exploreBranches(this.sibling, limit);
        double logForwardTotalScore = this.computeLogTotalScore();
        Node newSibling = this.drawCandidate();
        double logForwardBranchProbability = this.logScores[newSibling.getNr()] - logForwardTotalScore;

        double logLimitRatio = isFullExploration ? 0.0 : this.computeLogLimitRatio(newSibling);
        if (logLimitRatio == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        double newParentHeight = this.drawHeight(newSibling);
        if (!this.isInsideBranch(newSibling, newParentHeight)) {
            return Double.NEGATIVE_INFINITY;
        }
        double logForwardHeightDensity = this.computeLogHeightDensity(newSibling, newParentHeight);

        // the reverse move redraws the current states of P given its current neighbours

        Node oldSibling = this.sibling;
        double oldParentHeight = this.parent.getHeight();
        double logBackwardStateDensity = this.nodeStateSampler.computeLogDensity(this.parent, this.createEmptyEdit());

        this.regraft(newSibling, newParentHeight);
        double logForwardStateDensity = this.redrawParentStates();

        // the reverse move draws the old branch and height from the new attachment point, whose scores are cached

        this.sibling = newSibling;
        this.exploreBranches(newSibling, limit);
        double logBackwardTotalScore = this.computeLogTotalScore();
        double logBackwardBranchProbability = this.logScores[oldSibling.getNr()] - logBackwardTotalScore;
        double logBackwardHeightDensity = this.computeLogHeightDensity(oldSibling, oldParentHeight);
        double logBackwardPickProbability = this.subtreeSelector.computeLogProbability(x);

        return (logBackwardPickProbability - logForwardPickProbability)
                + logLimitRatio
                + (logBackwardBranchProbability - logForwardBranchProbability)
                + (logBackwardHeightDensity - logForwardHeightDensity)
                + (logBackwardStateDensity - logForwardStateDensity);
    }

    /** Starts a new stamp for the cached scores, which only need resetting once the counter runs out. */
    private void startProposal() {
        if (this.proposalStamp == Integer.MAX_VALUE) {
            Arrays.fill(this.scoreStamps, 0);
            Arrays.fill(this.xDifferenceStamps, 0);
            this.proposalStamp = 0;
        }
        this.proposalStamp++;
    }

    /**
     * Draws the limit of a local exploration: the number max(1, N) + j of sites at which X may
     * differ from both ends of a candidate, where N is that number at the current branch and j
     * is geometric with probability q^j (1 - q).
     */
    private int drawLimit() {
        int numAdditionalSteps = this.drawNumAdditionalSteps();
        this.scoreBranch(this.sibling);
        return (int) Math.min((long) Math.max(1, this.subtreeNumDifferences[this.sibling.getNr()]) + numAdditionalSteps, Integer.MAX_VALUE - 1);
    }

    private int drawNumAdditionalSteps() {
        if (this.windowProbability == 0.0) {
            return 0;
        }

        double numAdditionalSteps = Math.floor(Math.log(Randomizer.nextDouble()) / Math.log(this.windowProbability));
        return (int) Math.min(numAdditionalSteps, this.tree.getNodeCount());
    }

    /**
     * Returns the log ratio of the probabilities of drawing the limit of the reverse and of the
     * forward exploration. The reverse move has to draw the same limit from the new branch,
     * with j' = T - max(1, N'), which gives the factor q^(max(1, N) - max(1, N')). With q = 0,
     * a move that changes max(1, N) cannot be reversed.
     */
    private double computeLogLimitRatio(Node newSibling) {
        int oldMinLimit = Math.max(1, this.subtreeNumDifferences[this.sibling.getNr()]);
        int newMinLimit = Math.max(1, this.subtreeNumDifferences[newSibling.getNr()]);
        if (oldMinLimit == newMinLimit) {
            return 0.0;
        }
        return this.windowProbability == 0.0 ? Double.NEGATIVE_INFINITY : (oldMinLimit - newMinLimit) * Math.log(this.windowProbability);
    }

    /* Exploration */

    /**
     * Collects the candidate branches of the pruned tree around the branch above the given
     * node, breadth-first, and scores them. A step goes to a branch that meets the current one
     * at its top or at its bottom. Only branches whose top is older than X are candidates,
     * which form a connected part of the pruned tree around its root. The limit is the number
     * of sites at which X may differ from both ends of a candidate, where the exploration only
     * continues through candidates. The start must be within the limit. The neighbourhood is the
     * same from every candidate with the same limit.
     */
    private void exploreBranches(Node start, int limit) {
        if (this.explorationStamp == Integer.MAX_VALUE) {
            Arrays.fill(this.explorationStamps, 0);
            this.explorationStamp = 0;
        }
        this.explorationStamp++;
        this.numCandidates = 0;

        this.visitBranch(start, limit);
        for (int next = 0; next < this.numCandidates; next++) {
            Node node = this.candidates[next];

            // the branches meeting at the top, including the one above the top
            Node top = this.getPrunedParent(node);
            for (int childIndex = 0; childIndex < top.getChildCount(); childIndex++) {
                Node child = top.getChild(childIndex);
                this.visitBranch(this.getPrunedChild(child), limit);
            }
            if (!top.isRoot()) {
                this.visitBranch(top, limit);
            }

            // the branches meeting at the bottom
            for (int childIndex = 0; childIndex < node.getChildCount(); childIndex++) {
                Node child = node.getChild(childIndex);
                this.visitBranch(this.getPrunedChild(child), limit);
            }
        }
    }

    /**
     * Adds the branch above the given node as a candidate, unless it was reached before, lies
     * below X, or X differs from both of its ends at more sites than the given limit.
     */
    private void visitBranch(Node node, int limit) {
        if (this.explorationStamps[node.getNr()] == this.explorationStamp) {
            return;
        }
        this.explorationStamps[node.getNr()] = this.explorationStamp;

        if (this.getPrunedParent(node).getHeight() <= this.x.getHeight()) {
            return;
        }

        this.scoreBranch(node);
        if (this.subtreeNumDifferences[node.getNr()] > limit) {
            return;
        }

        this.candidates[this.numCandidates] = node;
        this.numCandidates++;
    }

    /** Scores the branch above the given node unless the current proposal already did. */
    private void scoreBranch(Node node) {
        if (this.scoreStamps[node.getNr()] != this.proposalStamp) {
            this.scoreStamps[node.getNr()] = this.proposalStamp;
            this.logScores[node.getNr()] = this.computeLogScore(node);
        }
    }

    /**
     * Computes the log score of attaching P at the midpoint of the branch above the given node
     * as the exact score does, but with the changes that a Jukes-Cantor model needs: each change
     * has probability d / 3 on a branch of distance d, and every site that does not change
     * costs e^-d. With P in the majority state of the top, the bottom and X, the branch is split
     * into the changes above P, those below it, and the sites at which X differs from P, which
     * attach to the new branch of P. All of these follow from the changes on the branch and the
     * state of X at their sites, as the number of differences of X to one end gives that to the other.
     */
    private double computeLogScore(Node node) {
        Node top = this.getPrunedParent(node);

        // the changes between the ends of the pruned branch, which includes P if it hangs above the node
        BranchChanges changes = node.getParent() == this.parent ? this.nodeStates.composeChanges(top, node) : this.nodeStates.getChanges(node);

        int numChanges = changes.size();
        int numTowardsX = 0;
        int numAwayFromX = 0;
        for (int i = 0; i < numChanges; i++) {
            int compatibleStates = this.xStates[changes.getSite(i)];
            boolean isFromCompatible = (compatibleStates & (1 << changes.getFrom(i))) != 0;
            boolean isToCompatible = (compatibleStates & (1 << changes.getTo(i))) != 0;
            if (isToCompatible && !isFromCompatible) {
                numTowardsX++;
            } else if (isFromCompatible && !isToCompatible) {
                numAwayFromX++;
            }
        }

        int numTopDifferences = this.computeNumDifferences(top, node, numAwayFromX - numTowardsX);
        this.upperNumDifferences[node.getNr()] = numTowardsX;
        this.lowerNumDifferences[node.getNr()] = numChanges - numTowardsX;
        this.subtreeNumDifferences[node.getNr()] = numTopDifferences - numTowardsX;

        double lowerHeight = Math.max(node.getHeight(), this.x.getHeight());
        double midHeight = 0.5 * (lowerHeight + top.getHeight());

        // the rate of a branch is that of the node below it, so the branch above P has the rate of P
        double upperDistance = this.substitutionProcess.computeDistance(this.parent, top.getHeight() - midHeight);
        double lowerDistance = this.substitutionProcess.computeDistance(node, midHeight - node.getHeight());
        double sideDistance = this.substitutionProcess.computeDistance(this.x, midHeight - this.x.getHeight());
        double untakenDistance = this.substitutionProcess.computeDistance(node, top.getHeight() - node.getHeight());

        double logRatio = -this.numSites * (upperDistance + lowerDistance + sideDistance - untakenDistance);
        logRatio += this.computeLogChanges(numTowardsX, upperDistance);
        logRatio += this.computeLogChanges(numChanges - numTowardsX, lowerDistance);
        logRatio += this.computeLogChanges(this.subtreeNumDifferences[node.getNr()], sideDistance);
        logRatio -= this.computeLogChanges(numChanges, untakenDistance);
        return this.annealing * logRatio;
    }

    /** Returns the log probability of the given number of changes at specific sites on a branch of the given distance under Jukes-Cantor. */
    private double computeLogChanges(int numChanges, double distance) {
        return numChanges == 0 ? 0.0 : numChanges * Math.log(distance / 3.0);
    }

    /**
     * Returns the number of sites at which X differs from the top of the pruned branch above the
     * given node, and records it and the number at the bottom. The numbers at the two ends differ by
     * the given change along the branch, and one of them is known if the exploration came from there,
     * or else it is counted along the path from the node to X.
     */
    private int computeNumDifferences(Node top, Node node, int numAddedDifferences) {
        if (this.xDifferenceStamps[top.getNr()] == this.proposalStamp) {
            this.recordNumDifferences(node, this.numXDifferences[top.getNr()] + numAddedDifferences);
            return this.numXDifferences[top.getNr()];
        }

        int numBottomDifferences;
        if (this.xDifferenceStamps[node.getNr()] == this.proposalStamp) {
            numBottomDifferences = this.numXDifferences[node.getNr()];
        } else {
            numBottomDifferences = this.countDifferencesToX(node);
        }

        this.recordNumDifferences(node, numBottomDifferences);
        this.recordNumDifferences(top, numBottomDifferences - numAddedDifferences);
        return numBottomDifferences - numAddedDifferences;
    }

    private void recordNumDifferences(Node node, int numDifferences) {
        this.numXDifferences[node.getNr()] = numDifferences;
        this.xDifferenceStamps[node.getNr()] = this.proposalStamp;
    }

    /** Counts the sites at which the given node is in a state that does not count as equal to X, along the path between them. */
    private int countDifferencesToX(Node node) {
        BranchChanges changes = this.nodeStates.composeChanges(node, this.x);

        int numDifferences = 0;
        for (int i = 0; i < changes.size(); i++) {
            numDifferences += (this.xStates[changes.getSite(i)] & (1 << changes.getFrom(i))) == 0 ? 1 : 0;
        }
        return numDifferences;
    }

    /**
     * Returns per site the bit mask of the states that count as equal to X. The states of a tip
     * at its missing sites, and at its ambiguous sites among the states compatible with its data,
     * are latent and not changed by the move, so they must not count as differences, which
     * would make the current neighbourhood look closer than the others. Inner nodes have no latent sites.
     */
    private int[] computeXStates() {
        // the materialized sequence is turned into masks in place
        int[] states = this.nodeStates.materializeState(this.x);
        for (int site = 0; site < this.numSites; site++) {
            states[site] = 1 << states[site];
        }

        for (int site : this.tipCompatibility.getMissingSites(this.x)) {
            states[site] = this.tipCompatibility.getEveryStateMask();
        }
        int[] ambiguousSites = this.tipCompatibility.getAmbiguousSites(this.x);
        int[] ambiguousStates = this.tipCompatibility.getAmbiguousStates(this.x);
        for (int i = 0; i < ambiguousSites.length; i++) {
            states[ambiguousSites[i]] = ambiguousStates[i];
        }
        return states;
    }

    /**
     * Returns log Σ exp(log score) over the candidates of the last exploration, and keeps the
     * terms of the sum, relative to the largest score, for drawing a candidate.
     */
    private double computeLogTotalScore() {
        double maxLogScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < this.numCandidates; i++) {
            maxLogScore = Math.max(maxLogScore, this.logScores[this.candidates[i].getNr()]);
        }

        double totalScore = 0.0;
        for (int i = 0; i < this.numCandidates; i++) {
            this.candidateScores[i] = Math.exp(this.logScores[this.candidates[i].getNr()] - maxLogScore);
            totalScore += this.candidateScores[i];
        }
        this.totalCandidateScore = totalScore;
        return maxLogScore + Math.log(totalScore);
    }

    /** Draws a candidate of the last exploration proportional to its score, which computeLogTotalScore has to have summed up. */
    private Node drawCandidate() {
        double remaining = Randomizer.nextDouble() * this.totalCandidateScore;
        for (int i = 0; i < this.numCandidates; i++) {
            remaining -= this.candidateScores[i];
            if (remaining < 0.0) {
                return this.candidates[i];
            }
        }

        // only rounding gets here
        return this.candidates[this.numCandidates - 1];
    }

    /* Height */

    /**
     * Draws the height of P on the branch above the given node within [max(t_C, t_X), t_D]. Its
     * relative position u from the bottom is Beta(f b + 1, f a + 1) with a and b the changes
     * above and below P if it took the majority state, from computeLogScore, which is the density of the position when a
     * changes fall above P and b below it uniformly along the branch.
     */
    private double drawHeight(Node node) {
        double alpha = this.annealing * this.lowerNumDifferences[node.getNr()] + 1.0;
        double beta = this.annealing * this.upperNumDifferences[node.getNr()] + 1.0;

        double lowerGamma = Randomizer.nextGamma(alpha, 1.0);
        double upperGamma = Randomizer.nextGamma(beta, 1.0);
        double position = lowerGamma / (lowerGamma + upperGamma);

        double lowerHeight = Math.max(node.getHeight(), this.x.getHeight());
        double upperHeight = this.getPrunedParent(node).getHeight();
        return lowerHeight + position * (upperHeight - lowerHeight);
    }

    /** Computes the log density with which drawHeight draws the given height on the branch above the given node, which must have been scored. */
    private double computeLogHeightDensity(Node node, double height) {
        double alpha = this.annealing * this.lowerNumDifferences[node.getNr()] + 1.0;
        double beta = this.annealing * this.upperNumDifferences[node.getNr()] + 1.0;

        double lowerHeight = Math.max(node.getHeight(), this.x.getHeight());
        double upperHeight = this.getPrunedParent(node).getHeight();
        double position = (height - lowerHeight) / (upperHeight - lowerHeight);

        return BetaDistribution.of(alpha, beta).logDensity(position) - Math.log(upperHeight - lowerHeight);
    }

    /** Checks that the given height lies strictly inside the part of the branch above the given node that is older than X. */
    private boolean isInsideBranch(Node node, double height) {
        return height > node.getHeight() && height > this.x.getHeight() && height < this.getPrunedParent(node).getHeight();
    }

    /* Tree Rearrangement */

    /**
     * Moves P with the subtree below X onto the branch above the given node at the given
     * height, and rewrites the changes of the branches whose parent changes, which keeps all
     * node states. If the node is the current sibling, only the height of P changes.
     */
    private void regraft(Node newSibling, double newParentHeight) {
        if (newSibling == this.sibling) {
            this.parent.setHeight(newParentHeight);
            return;
        }

        Node grandparent = this.parent.getParent();
        Node newGrandparent = this.getPrunedParent(newSibling);

        // the new changes are composed along paths of the tree before the move
        BranchChanges newParentChanges = this.nodeStates.composeChanges(newGrandparent, this.parent);
        BranchChanges newSiblingChanges = this.nodeStates.composeChanges(this.parent, newSibling);
        BranchChanges siblingChanges = this.nodeStates.composeChanges(grandparent, this.sibling);

        this.replace(grandparent, this.parent, this.sibling);
        this.replace(this.parent, this.sibling, newSibling);
        this.replace(newGrandparent, newSibling, this.parent);
        this.parent.setHeight(newParentHeight);

        // the counts of a branch use the state counts of its parent, so P comes first
        this.nodeStates.setBranchChanges(this.parent, newParentChanges, this);
        this.nodeStates.setBranchChanges(newSibling, newSiblingChanges, this);
        this.nodeStates.setBranchChanges(this.sibling, siblingChanges, this);
    }

    /** Returns an edit that keeps the current states of P, whose density is that of drawing them. */
    private StateEdit createEmptyEdit() {
        return new StateEdit(this.parent.getNr(), new int[0], new int[0]);
    }

    /**
     * Draws the states of P from their full conditional given the states of its parent and
     * children, applies them, and returns the log density with which they were drawn.
     */
    private double redrawParentStates() {
        StateEdit edit = this.nodeStateSampler.sample(this.parent);
        double logDensity = this.nodeStateSampler.computeLogDensity(this.parent, edit);
        this.nodeStates.setStates(this.parent, edit.sites(), edit.states(), this);
        return logDensity;
    }

    /** Returns the parent of the given node in the pruned tree, where S hangs directly below G. */
    private Node getPrunedParent(Node node) {
        return node.getParent() == this.parent ? this.parent.getParent() : node.getParent();
    }

    /** Returns the given child in the pruned tree, where S takes the place of P. */
    private Node getPrunedChild(Node child) {
        return child == this.parent ? this.sibling : child;
    }

}
