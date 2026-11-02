package emat.operators.nodetimes.batch;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import emat.operators.nodetimes.NodeTimes;
import emat.operators.nodetimes.batch.TruncatedDensityTransport.Tails;

@Description("Moves half of the internal nodes together along one collective coordinate of their normal scores. The nodes " +
        "are those at even or at odd depth below the root, so that none of them is the parent of another one and the range " +
        "and the conditional density of each node stay the same while the others move. The conditional density of the height " +
        "of a node (the genetic prior times a learned quadratic approximation of the tree prior) is transported to a " +
        "standard normal score. The sum t of the scores, divided by the square root of their number, is standard normal, too. " +
        "It is rotated with an auxiliary momentum p, (t, p) -> (t cos a + p sin a, p cos a - t sin a), which leaves the " +
        "standard normal distribution of both invariant, and every score moves by the same share of the change of t. " +
        "The momentum persists from one proposal to the next: it is kept after an accepted proposal and flipped after a " +
        "rejected one, and it is partially refreshed before each proposal (generalised Hamiltonian Monte Carlo). The chain " +
        "is non-reversible, leaves the posterior invariant, and moves the tree length coherently over many proposals. The " +
        "move is accepted unless the tree prior differs from its approximation. It changes only the collective coordinate " +
        "and leaves the other components of the scores as they are, so it has to be used next to operators that move the " +
        "nodes individually.")
public class GaussianRotationNodeDisplacementOperator extends TransportedNodeDisplacementOperator {

    final public Input<Double> refreshInput = new Input<>("refresh", "the share in (0, 1] of the momentum that is drawn anew before each proposal. A refresh of one gives an independent momentum and a reversible move; a small refresh keeps the direction for many proposals", 0.2);

    // a quarter turn swaps the collective coordinate and the momentum, and a larger angle is equivalent to a smaller one
    private static final double MAX_STEP_SIZE = Math.PI / 2.0;

    private double refresh;

    // the momentum between the proposals, and the one of the last proposal before and after the rotation
    private double momentum;
    private double usedMomentum;
    private double rotatedMomentum;

    // the nodes of the last proposal that had room, with their conditionals and normal scores
    private Node[] batchNodes;
    private Conditional[] batchConditionals;
    private double[] batchScores;

    // scratch space of the walk through the tree
    private int[] depths;
    private Node[] stack;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.refresh = this.refreshInput.get();
        if (this.refresh <= 0.0 || this.refresh > 1.0) {
            throw new IllegalArgumentException("The refresh must lie in (0, 1].");
        }

        this.momentum = Randomizer.nextGaussian();

        int numNodes = this.tree.getNodeCount();
        this.batchNodes = new Node[numNodes];
        this.batchConditionals = new Conditional[numNodes];
        this.batchScores = new double[numNodes];
        this.depths = new int[numNodes];
        this.stack = new Node[numNodes];
    }

    /**
     * Rotates the collective coordinate of the normal scores of the nodes at even or at odd depth, with equal
     * probability. The depth parity of a node does not change in the move, so the reverse move picks the same nodes.
     * They are no neighbours, so the ranges and the conditionals of the nodes with room are the same in both directions.
     */
    @Override
    protected double moveNodes() {
        int parity = Randomizer.nextBoolean() ? 1 : 0;
        int numNodes = this.collectNodesWithRoom(parity);

        this.usedMomentum = this.refreshMomentum();
        this.rotatedMomentum = this.usedMomentum;
        if (numNodes == 0) {
            return Double.NEGATIVE_INFINITY;
        }

        // the collective coordinate is standard normal when the nodes follow their conditionals
        double sqrtNumNodes = Math.sqrt(numNodes);
        double coordinate = 0.0;
        for (int i = 0; i < numNodes; i++) {
            coordinate += this.batchScores[i];
        }
        coordinate /= sqrtNumNodes;

        double angle = this.getCoercableParameterValue();
        double newCoordinate = coordinate * Math.cos(angle) + this.usedMomentum * Math.sin(angle);
        this.rotatedMomentum = this.usedMomentum * Math.cos(angle) - coordinate * Math.sin(angle);

        double scoreShift = (newCoordinate - coordinate) / sqrtNumNodes;

        double logHastingsRatio = 0.0;
        for (int i = 0; i < numNodes; i++) {
            Conditional conditional = this.batchConditionals[i];
            double newHeight = conditional.computeHeight(Tails.ofNormalScore(this.batchScores[i] + scoreShift));
            logHastingsRatio += this.displace(this.batchNodes[i], conditional, newHeight);
        }

        return logHastingsRatio;
    }

    /**
     * Collects the internal nodes below the root with the given depth parity that have room for a move, together
     * with their conditionals and normal scores, and returns their number. The range of such a node is bounded by its
     * parent and children, which have the other parity and keep their heights.
     */
    private int collectNodesWithRoom(int parity) {
        int numNodes = 0;

        int stackSize = 0;
        Node root = this.tree.getRoot();
        this.depths[root.getNr()] = 0;
        this.stack[stackSize++] = root;

        while (stackSize > 0) {
            Node node = this.stack[--stackSize];
            int depth = this.depths[node.getNr()];

            if (!node.isLeaf() && !node.isRoot() && depth % 2 == parity) {
                double minHeight = NodeTimes.computeMinHeight(this.mutations, node);
                double maxHeight = NodeTimes.computeMaxHeight(this.mutations, node);

                if (maxHeight > minHeight) {
                    Conditional conditional = this.createConditional(node, minHeight, maxHeight);
                    this.batchNodes[numNodes] = node;
                    this.batchConditionals[numNodes] = conditional;
                    this.batchScores[numNodes] = conditional.computeTails(node.getHeight()).toNormalScore();
                    numNodes++;
                }
            }

            for (Node child : node.getChildren()) {
                this.depths[child.getNr()] = depth + 1;
                this.stack[stackSize++] = child;
            }
        }

        return numNodes;
    }

    /** Returns the momentum with a share of it drawn anew, which keeps the standard normal distribution. */
    private double refreshMomentum() {
        return Math.sqrt(1.0 - this.refresh * this.refresh) * this.momentum + this.refresh * Randomizer.nextGaussian();
    }

    /** Keeps the rotated momentum after an accepted proposal, which continues the movement in the same direction. */
    @Override
    public void accept() {
        super.accept();
        this.momentum = this.rotatedMomentum;
    }

    /** Reverses the momentum after a rejected proposal, whatever the reason of the rejection. */
    @Override
    public void reject(int reason) {
        super.reject(reason);
        this.momentum = -this.usedMomentum;
    }

    @Override
    protected double getMaxStepSize() {
        return MAX_STEP_SIZE;
    }

}
