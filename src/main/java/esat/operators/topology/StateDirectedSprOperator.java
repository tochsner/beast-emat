package esat.operators.topology;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import emat.operators.topology.mdspr.BranchLengthSubtreeSelector;
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
        "All other node states are kept. Each branch of the pruned tree is scored by the genetic prior of attaching " +
        "the parent at its midpoint, summed over the states of the parent, relative to not attaching it there, which " +
        "makes the choice of the branch an approximate Gibbs step that does not depend on the current states of the " +
        "parent. Most proposals only consider the branches in a neighbourhood of the current attachment " +
        "point, either within a radius or through branches at which the subtree differs from both ends at few sites. " +
        "The height of the parent on the new branch is drawn from a Beta distribution given by the changes " +
        "above and below the parent if it took the majority state of its three neighbours. The root is never changed.")
public class StateDirectedSprOperator extends TreeOperator {

    public enum Neighbourhood {
        RADIUS,
        DIFFERENCES
    }

    final public Input<NodeStates> nodeStatesInput = new Input<>("nodeStates", "the node states, of which the move redraws those of the parent of the pruned subtree", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<Double> fullExplorationProbabilityInput = new Input<>("fullExplorationProbability", "the probability of considering every branch instead of the neighbourhood of the current attachment point", 0.01);
    final public Input<Neighbourhood> neighbourhoodInput = new Input<>("neighbourhood", "the neighbourhood of the current attachment point that a local proposal considers: RADIUS takes the branches " +
            "within 1 + j steps between branches that meet at a node, DIFFERENCES the branches connected to the current one through branches at which X differs from both ends at no more than " +
            "max(1, N) + j sites, where N is that number at the current branch", Neighbourhood.RADIUS, Neighbourhood.values());
    final public Input<Double> windowProbabilityInput = new Input<>("windowProbability", "the parameter q of the size of the neighbourhood, which grows by j geometric with probability q^j (1 - q)", 0.5);
    final public Input<Double> annealingInput = new Input<>("annealing", "the power f to which the scores of the branches and the density of the height are raised, which makes the proposals less eager", 0.8);
    final public Input<Boolean> scaleWeightBySqrtNumTipsInput = new Input<>("scaleWeightBySqrtNumTips", "whether to scale the weight by the square root of the number of tips n, so that the operator is selected with weight * sqrt(n). Otherwise, the weight is used as given", false);

    Tree tree;
    NodeStates nodeStates;
    SubstitutionProcess substitutionProcess;
    NodeStateSampler nodeStateSampler;
    BranchLengthSubtreeSelector subtreeSelector;

    int numStates;
    Neighbourhood neighbourhood;
    double fullExplorationProbability;
    double windowProbability;
    double annealing;

    // the factor sqrt(n) or one by which the weight of the operator is scaled
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

    // per node: the exploration that reached the branch above it, and the reached branches in the order of their distance with that distance
    int explorationStamp;
    int[] explorationStamps;
    Node[] candidates;
    int[] candidateDistances;
    int numCandidates;

    // scratch space for the number of sites per triple of states of the top, the bottom and X, and for the transition probabilities
    int[] tripleCounts;
    double[] upperProbabilities;
    double[] lowerProbabilities;
    double[] sideProbabilities;
    double[] untakenProbabilities;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.nodeStates = this.nodeStatesInput.get();
        this.substitutionProcess = new SubstitutionProcess(this.geneticPriorInput.get());
        this.nodeStateSampler = new NodeStateSampler(this.nodeStates, this.substitutionProcess);
        this.subtreeSelector = new BranchLengthSubtreeSelector(this.tree);

        this.numStates = this.nodeStates.getNumStates();
        this.neighbourhood = this.neighbourhoodInput.get();
        this.fullExplorationProbability = this.fullExplorationProbabilityInput.get();
        this.windowProbability = this.windowProbabilityInput.get();
        this.annealing = this.annealingInput.get();
        if (this.windowProbability < 0.0 || this.windowProbability >= 1.0) {
            throw new IllegalArgumentException("windowProbability must be at least 0 and less than 1.");
        }
        if (this.annealing <= 0.0) {
            throw new IllegalArgumentException("annealing must be positive.");
        }

        this.weightFactor = this.scaleWeightBySqrtNumTipsInput.get() ? Math.sqrt(this.tree.getLeafNodeCount()) : 1.0;

        int numNodes = this.tree.getNodeCount();
        this.scoreStamps = new int[numNodes];
        this.logScores = new double[numNodes];
        this.upperNumDifferences = new int[numNodes];
        this.lowerNumDifferences = new int[numNodes];
        this.subtreeNumDifferences = new int[numNodes];
        this.explorationStamps = new int[numNodes];
        this.candidates = new Node[numNodes];
        this.candidateDistances = new int[numNodes];

        this.tripleCounts = new int[this.numStates * this.numStates * this.numStates];
        this.upperProbabilities = new double[this.numStates * this.numStates];
        this.lowerProbabilities = new double[this.numStates * this.numStates];
        this.sideProbabilities = new double[this.numStates * this.numStates];
        this.untakenProbabilities = new double[this.numStates * this.numStates];
    }

    /** Returns the weight, scaled by the square root of the number of tips if requested. */
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
     * the candidates around the new one with the same limit and the same scores. Within a
     * radius, these include the current branch as the radius is a symmetric distance, and a
     * difference neighbourhood is the same from all of its branches. The reverse move redraws
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

        boolean isFullExploration = Randomizer.nextDouble() < this.fullExplorationProbability;
        int limit = isFullExploration ? Integer.MAX_VALUE : this.drawLimit();

        // draw the new branch and the new height of P on it

        this.exploreBranches(this.sibling, limit);
        double logForwardTotalScore = this.computeLogTotalScore();
        Node newSibling = this.drawCandidate(logForwardTotalScore);
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
            this.proposalStamp = 0;
        }
        this.proposalStamp++;
    }

    /**
     * Draws the limit of a local exploration: the radius 1 + j, or the number max(1, N) + j of
     * sites at which X may differ from both ends of a candidate, where N is that number at the
     * current branch. In both, j is geometric with probability q^j (1 - q).
     */
    private int drawLimit() {
        int numAdditionalSteps = this.drawNumAdditionalSteps();
        if (this.neighbourhood == Neighbourhood.RADIUS) {
            return 1 + numAdditionalSteps;
        }

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
     * forward exploration. The radius is drawn independently of the attachment point, but the
     * reverse move has to draw the same difference limit from the new branch, with j' = T -
     * max(1, N'), which gives the factor q^(max(1, N) - max(1, N')). With q = 0, a move that
     * changes max(1, N) cannot be reversed.
     */
    private double computeLogLimitRatio(Node newSibling) {
        if (this.neighbourhood == Neighbourhood.RADIUS) {
            return 0.0;
        }

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
     * which form a connected part of the pruned tree around its root. The limit is the radius
     * in steps, or the number of sites at which X may differ from both ends of a candidate,
     * where the exploration only continues through candidates. The start must be within the
     * limit. Both neighbourhoods are the same from every candidate with the same limit.
     */
    private void exploreBranches(Node start, int limit) {
        if (this.explorationStamp == Integer.MAX_VALUE) {
            Arrays.fill(this.explorationStamps, 0);
            this.explorationStamp = 0;
        }
        this.explorationStamp++;
        this.numCandidates = 0;

        this.visitBranch(start, 0, limit);
        for (int next = 0; next < this.numCandidates; next++) {
            Node node = this.candidates[next];
            int distance = this.candidateDistances[next];
            if (this.neighbourhood == Neighbourhood.RADIUS && distance >= limit) {
                continue;
            }

            // the branches meeting at the top, including the one above the top
            Node top = this.getPrunedParent(node);
            for (Node child : top.getChildren()) {
                this.visitBranch(this.getPrunedChild(child), distance + 1, limit);
            }
            if (!top.isRoot()) {
                this.visitBranch(top, distance + 1, limit);
            }

            // the branches meeting at the bottom
            for (Node child : node.getChildren()) {
                this.visitBranch(this.getPrunedChild(child), distance + 1, limit);
            }
        }
    }

    /**
     * Adds the branch above the given node as a candidate at the given distance, unless it was
     * reached before, lies below X, or X differs from both of its ends at more sites than the
     * given limit of a difference neighbourhood.
     */
    private void visitBranch(Node node, int distance, int limit) {
        if (this.explorationStamps[node.getNr()] == this.explorationStamp) {
            return;
        }
        this.explorationStamps[node.getNr()] = this.explorationStamp;

        if (this.getPrunedParent(node).getHeight() <= this.x.getHeight()) {
            return;
        }

        this.scoreBranch(node);
        if (this.neighbourhood == Neighbourhood.DIFFERENCES && this.subtreeNumDifferences[node.getNr()] > limit) {
            return;
        }

        this.candidates[this.numCandidates] = node;
        this.candidateDistances[this.numCandidates] = distance;
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
     * of the pruned tree, between its top D and its bottom C. The genetic prior of the pruned
     * tree with P attached there, summed over the states of P, is the product over all
     * branches of the pruned tree, with the term of D–C replaced by Π_sites Σ_p P_DP(d, p)
     * P_PC(p, c) P_PX(p, x). The product over all branches is the same for every candidate, so
     * the score is the ratio of the replacing term to the replaced one, raised to the
     * annealing power. The sites only enter through the number of sites in each triple of
     * states (d, c, x), so the score does not depend on the states of P.
     */
    private double computeLogScore(Node node) {
        Node top = this.getPrunedParent(node);
        double lowerHeight = Math.max(node.getHeight(), this.x.getHeight());
        double midHeight = 0.5 * (lowerHeight + top.getHeight());

        this.nodeStates.countTriples(top, node, this.x, this.tripleCounts);
        this.countMajorityChanges(node);

        // the rate of a branch is that of the node below it, so the branch above P has the rate of P
        this.substitutionProcess.getTransitionProbabilities(this.parent, top.getHeight() - midHeight, this.upperProbabilities);
        this.substitutionProcess.getTransitionProbabilities(node, midHeight - node.getHeight(), this.lowerProbabilities);
        this.substitutionProcess.getTransitionProbabilities(this.x, midHeight - this.x.getHeight(), this.sideProbabilities);
        this.substitutionProcess.getTransitionProbabilities(node, top.getHeight() - node.getHeight(), this.untakenProbabilities);

        int k = this.numStates;
        double logRatio = 0.0;
        for (int topState = 0; topState < k; topState++) {
            for (int bottomState = 0; bottomState < k; bottomState++) {
                for (int sideState = 0; sideState < k; sideState++) {
                    int count = this.tripleCounts[(topState * k + bottomState) * k + sideState];
                    if (count == 0) {
                        continue;
                    }

                    double takenProbability = 0.0;
                    for (int parentState = 0; parentState < k; parentState++) {
                        takenProbability += this.upperProbabilities[topState * k + parentState]
                                * this.lowerProbabilities[parentState * k + bottomState]
                                * this.sideProbabilities[parentState * k + sideState];
                    }
                    logRatio += count * (Math.log(takenProbability) - Math.log(this.untakenProbabilities[topState * k + bottomState]));
                }
            }
        }

        return this.annealing * logRatio;
    }

    /**
     * Counts the changes above and below P on the branch above the given node if P took the
     * majority state of the top, the bottom and X at every site, or the state of the top if
     * all three differ. These shape the density of the height of P. It also counts the sites at
     * which X differs from both ends, which bound the difference neighbourhood.
     */
    private void countMajorityChanges(Node node) {
        int k = this.numStates;
        int numUpperChanges = 0;
        int numLowerChanges = 0;
        int numSubtreeDifferences = 0;

        for (int topState = 0; topState < k; topState++) {
            for (int bottomState = 0; bottomState < k; bottomState++) {
                for (int sideState = 0; sideState < k; sideState++) {
                    int count = this.tripleCounts[(topState * k + bottomState) * k + sideState];
                    int parentState = bottomState == sideState ? bottomState : topState;
                    if (topState != parentState) {
                        numUpperChanges += count;
                    }
                    if (parentState != bottomState) {
                        numLowerChanges += count;
                    }
                    if (sideState != topState && sideState != bottomState) {
                        numSubtreeDifferences += count;
                    }
                }
            }
        }

        this.upperNumDifferences[node.getNr()] = numUpperChanges;
        this.lowerNumDifferences[node.getNr()] = numLowerChanges;
        this.subtreeNumDifferences[node.getNr()] = numSubtreeDifferences;
    }

    /** Returns log Σ exp(log score) over the candidates of the last exploration. */
    private double computeLogTotalScore() {
        double maxLogScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < this.numCandidates; i++) {
            maxLogScore = Math.max(maxLogScore, this.logScores[this.candidates[i].getNr()]);
        }

        double totalScore = 0.0;
        for (int i = 0; i < this.numCandidates; i++) {
            totalScore += Math.exp(this.logScores[this.candidates[i].getNr()] - maxLogScore);
        }
        return maxLogScore + Math.log(totalScore);
    }

    /** Draws a candidate of the last exploration proportional to its score. */
    private Node drawCandidate(double logTotalScore) {
        double remaining = Randomizer.nextDouble();
        for (int i = 0; i < this.numCandidates; i++) {
            remaining -= Math.exp(this.logScores[this.candidates[i].getNr()] - logTotalScore);
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
     * above and below P from countMajorityChanges, which is the density of the position when a
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
