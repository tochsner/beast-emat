package esat.operators.topology;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import emat.operators.topology.mdspr.BranchLengthSubtreeSelector;
import esat.prior.GeneticPrior;
import esat.state.NodeStates;
import esat.stochasticmapping.SubstitutionProcess;
import org.apache.commons.statistics.distribution.BetaDistribution;

import java.util.Arrays;

@Description("Regrafts a subtree, picked proportional to the length of the branch above it, together with its parent " +
        "onto another branch, keeping all node states. Each branch of the pruned tree is scored by the genetic prior " +
        "of attaching the parent at its midpoint relative to not attaching it there, which makes the choice of the " +
        "branch an approximate Gibbs step. Most proposals only consider the branches within a geometric number of " +
        "steps of the current attachment point. The height of the parent on the new branch is drawn from a Beta " +
        "distribution given by the differences of its states to the states at both ends. The root is never changed.")
public class StateDirectedSprOperator extends TreeOperator {

    final public Input<NodeStates> nodeStatesInput = new Input<>("nodeStates", "the node states, which the move keeps", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<Double> fullExplorationProbabilityInput = new Input<>("fullExplorationProbability", "the probability of considering every branch instead of the neighbourhood of the current attachment point", 0.01);
    final public Input<Double> windowProbabilityInput = new Input<>("windowProbability", "the parameter q of the radius 1 + j of the neighbourhood, with j geometric with probability q^j (1 - q). " +
            "The radius counts the steps between branches that meet at a node", 0.5);
    final public Input<Double> annealingInput = new Input<>("annealing", "the power f to which the scores of the branches and the density of the height are raised, which makes the proposals less eager", 0.8);
    final public Input<Boolean> scaleWeightBySqrtNumTipsInput = new Input<>("scaleWeightBySqrtNumTips", "whether to scale the weight by the square root of the number of tips n, so that the operator is selected with weight * sqrt(n). Otherwise, the weight is used as given", false);

    Tree tree;
    NodeStates nodeStates;
    SubstitutionProcess substitutionProcess;
    BranchLengthSubtreeSelector subtreeSelector;

    int numStates;
    double fullExplorationProbability;
    double windowProbability;
    double annealing;

    // the factor sqrt(n) or one by which the weight of the operator is scaled
    double weightFactor;

    // the subtree root X of the current proposal, its parent P that moves with it, and its sibling S
    Node x;
    Node parent;
    Node sibling;

    // per node: the proposal that scored the branch above it, its log score, and the differences of P to the top and to the bottom of the branch
    int proposalStamp;
    int[] scoreStamps;
    double[] logScores;
    int[] upperNumDifferences;
    int[] lowerNumDifferences;

    // per node: the exploration that reached the branch above it, and the reached branches in the order of their distance with that distance
    int explorationStamp;
    int[] explorationStamps;
    Node[] candidates;
    int[] candidateDistances;
    int numCandidates;

    // scratch space for transition counts and probabilities
    int[] counts;
    double[] transitionProbabilities;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.nodeStates = this.nodeStatesInput.get();
        this.substitutionProcess = new SubstitutionProcess(this.geneticPriorInput.get());
        this.subtreeSelector = new BranchLengthSubtreeSelector(this.tree);

        this.numStates = this.nodeStates.getNumStates();
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
        this.explorationStamps = new int[numNodes];
        this.candidates = new Node[numNodes];
        this.candidateDistances = new int[numNodes];

        this.counts = new int[this.numStates * this.numStates];
        this.transitionProbabilities = new double[this.numStates * this.numStates];
    }

    /** Returns the weight, scaled by the square root of the number of tips if requested. */
    @Override
    public double getWeight() {
        return super.getWeight() * this.weightFactor;
    }

    /**
     * Moves P with the subtree below X onto a branch of the pruned tree, i.e. the tree
     * without X and P in which G–P–S is joined into G–S. The branch is drawn proportional to
     * its score among the candidates, and the height of P on it from a Beta distribution. The
     * pruned tree and therefore the scores are the same before and after the move, so the
     * reverse move draws the current branch from the candidates around the new one with the
     * same radius. As the radius is a symmetric distance, they include the current branch.
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

        int radius = Randomizer.nextDouble() < this.fullExplorationProbability ? Integer.MAX_VALUE : this.drawRadius();

        // draw the new branch and the new height of P on it

        this.exploreBranches(this.sibling, radius);
        double logForwardTotalScore = this.computeLogTotalScore();
        Node newSibling = this.drawCandidate(logForwardTotalScore);
        double logForwardBranchProbability = this.logScores[newSibling.getNr()] - logForwardTotalScore;

        double newParentHeight = this.drawHeight(newSibling);
        if (!this.isInsideBranch(newSibling, newParentHeight)) {
            return Double.NEGATIVE_INFINITY;
        }
        double logForwardHeightDensity = this.computeLogHeightDensity(newSibling, newParentHeight);

        // the reverse move draws the current branch and height from the new attachment point

        this.exploreBranches(newSibling, radius);
        double logBackwardTotalScore = this.computeLogTotalScore();
        double logBackwardBranchProbability = this.logScores[this.sibling.getNr()] - logBackwardTotalScore;
        double logBackwardHeightDensity = this.computeLogHeightDensity(this.sibling, this.parent.getHeight());

        this.regraft(newSibling, newParentHeight);
        double logBackwardPickProbability = this.subtreeSelector.computeLogProbability(x);

        return (logBackwardPickProbability - logForwardPickProbability)
                + (logBackwardBranchProbability - logForwardBranchProbability)
                + (logBackwardHeightDensity - logForwardHeightDensity);
    }

    /** Starts a new stamp for the cached scores, which only need resetting once the counter runs out. */
    private void startProposal() {
        if (this.proposalStamp == Integer.MAX_VALUE) {
            Arrays.fill(this.scoreStamps, 0);
            this.proposalStamp = 0;
        }
        this.proposalStamp++;
    }

    /** Draws the radius 1 + j of a local exploration, with j geometric with probability q^j (1 - q). */
    private int drawRadius() {
        if (this.windowProbability == 0.0) {
            return 1;
        }

        double numAdditionalSteps = Math.floor(Math.log(Randomizer.nextDouble()) / Math.log(this.windowProbability));
        return 1 + (int) Math.min(numAdditionalSteps, this.tree.getNodeCount());
    }

    /* Exploration */

    /**
     * Collects the candidate branches of the pruned tree within the given number of steps of
     * the branch above the given node, breadth-first, and scores them. A step goes to a branch
     * that meets the current one at its top or at its bottom. Only branches whose top is older
     * than X are candidates, which form a connected part of the pruned tree around its root.
     */
    private void exploreBranches(Node start, int radius) {
        if (this.explorationStamp == Integer.MAX_VALUE) {
            Arrays.fill(this.explorationStamps, 0);
            this.explorationStamp = 0;
        }
        this.explorationStamp++;
        this.numCandidates = 0;

        this.visitBranch(start, 0);
        for (int next = 0; next < this.numCandidates; next++) {
            Node node = this.candidates[next];
            int distance = this.candidateDistances[next];
            if (distance >= radius) {
                continue;
            }

            // the branches meeting at the top, including the one above the top
            Node top = this.getPrunedParent(node);
            for (Node child : top.getChildren()) {
                this.visitBranch(this.getPrunedChild(child), distance + 1);
            }
            if (!top.isRoot()) {
                this.visitBranch(top, distance + 1);
            }

            // the branches meeting at the bottom
            for (Node child : node.getChildren()) {
                this.visitBranch(this.getPrunedChild(child), distance + 1);
            }
        }
    }

    /** Adds the branch above the given node as a candidate at the given distance, unless it was reached before or lies below X. */
    private void visitBranch(Node node, int distance) {
        if (this.explorationStamps[node.getNr()] == this.explorationStamp) {
            return;
        }
        this.explorationStamps[node.getNr()] = this.explorationStamp;

        if (this.getPrunedParent(node).getHeight() <= this.x.getHeight()) {
            return;
        }

        this.candidates[this.numCandidates] = node;
        this.candidateDistances[this.numCandidates] = distance;
        this.numCandidates++;

        if (this.scoreStamps[node.getNr()] != this.proposalStamp) {
            this.scoreStamps[node.getNr()] = this.proposalStamp;
            this.logScores[node.getNr()] = this.computeLogScore(node);
        }
    }

    /**
     * Computes the log score of attaching P at the midpoint of the branch above the given node
     * of the pruned tree, between its top D and its bottom C. The genetic prior of the pruned
     * tree with P attached there is the product over all branches of the pruned tree, with the
     * term of D–C replaced by those of D–P, P–C and P–X. The product over all branches is the
     * same for every candidate, so the score is the ratio of the replacing terms to the
     * replaced one, raised to the annealing power. It also records the differences of P to D
     * and to C, which shape the density of the height.
     */
    private double computeLogScore(Node node) {
        Node top = this.getPrunedParent(node);
        double lowerHeight = Math.max(node.getHeight(), this.x.getHeight());
        double midHeight = 0.5 * (lowerHeight + top.getHeight());

        this.nodeStates.countTransitions(top, this.parent, this.counts);
        this.upperNumDifferences[node.getNr()] = this.countDifferences(this.counts);
        double logTakenProbability = this.computeLogBranchProbability(this.counts, this.parent, top.getHeight() - midHeight);

        this.nodeStates.countTransitions(this.parent, node, this.counts);
        this.lowerNumDifferences[node.getNr()] = this.countDifferences(this.counts);
        logTakenProbability += this.computeLogBranchProbability(this.counts, node, midHeight - node.getHeight());

        logTakenProbability += this.computeLogBranchProbability(
                this.nodeStates.getTransitionCounts(this.x), this.x, midHeight - this.x.getHeight()
        );

        // the branch above S joins the branches above P and S, so its counts are not in the node states
        int[] untakenCounts = this.nodeStates.getTransitionCounts(node);
        if (node == this.sibling) {
            this.nodeStates.countTransitions(top, node, this.counts);
            untakenCounts = this.counts;
        }
        double logUntakenProbability = this.computeLogBranchProbability(untakenCounts, node, top.getHeight() - node.getHeight());

        return this.annealing * (logTakenProbability - logUntakenProbability);
    }

    /** Computes Σ N_ab log P_ab for the given counts on a branch of the given length at the rate of the branch above the given node. */
    private double computeLogBranchProbability(int[] counts, Node rateNode, double branchLength) {
        this.substitutionProcess.getTransitionProbabilities(rateNode, branchLength, this.transitionProbabilities);

        double logProbability = 0.0;
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] != 0) {
                logProbability += counts[i] * Math.log(this.transitionProbabilities[i]);
            }
        }
        return logProbability;
    }

    private int countDifferences(int[] counts) {
        int numDifferences = 0;
        for (int from = 0; from < this.numStates; from++) {
            for (int to = 0; to < this.numStates; to++) {
                if (from != to) {
                    numDifferences += counts[from * this.numStates + to];
                }
            }
        }
        return numDifferences;
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
     * relative position u from the bottom is Beta(f b + 1, f a + 1) with a and b the differences
     * of P to the top and to the bottom, which is the density of the position when a changes
     * fall above P and b below it uniformly along the branch.
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
     * height, and recomputes the transition counts of the branches whose parent changes. If
     * the node is the current sibling, only the height of P changes.
     */
    private void regraft(Node newSibling, double newParentHeight) {
        if (newSibling != this.sibling) {
            Node grandparent = this.parent.getParent();
            Node newGrandparent = this.getPrunedParent(newSibling);

            this.replace(grandparent, this.parent, this.sibling);
            this.replace(this.parent, this.sibling, newSibling);
            this.replace(newGrandparent, newSibling, this.parent);
        }
        this.parent.setHeight(newParentHeight);

        if (newSibling != this.sibling) {
            this.nodeStates.markBranchChanged(this.parent, this);
            this.nodeStates.markBranchChanged(newSibling, this);
            this.nodeStates.markBranchChanged(this.sibling, this);
        }
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
