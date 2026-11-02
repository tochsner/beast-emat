package emat.operators.nodetimes.batch;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.Operator;
import beast.base.inference.OperatorSchedule;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.Frequencies;
import beast.base.spec.evolution.substitutionmodel.HKY;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.SimplexParam;
import beast.base.util.Randomizer;
import emat.operators.nodetimes.height.NodeHeightMover;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MappedBatchNodeDisplacementOperatorTest {

    // nucleotide states
    private static final int A = 0;
    private static final int C = 1;
    private static final int G = 2;

    /** The tree, its mutations and the genetic prior of a test chain. */
    private record Chain(TreeParser tree, Mutations mutations, GeneticPrior geneticPrior) {
    }

    /**
     * Under HKY, whose total mutation rate depends on the state, moving a node and mapping the mutations on the three
     * branches around it changes the genetic prior by -k Δh plus the sum over the branches of m Δlog L, minus the
     * Jacobian of the map of the mutations. The decay rate k and the hazards come from the genetic prior as of before
     * the move, so this checks the closed form of the genetic prior that the operator draws from.
     */
    @Test
    public void testGeneticPriorOfMappedMutationsIsExact() {
        Randomizer.setSeed(3);
        Chain chain = this.createThreeTipChain(true);
        TreeParser tree = chain.tree();
        GeneticPrior geneticPrior = chain.geneticPrior();
        Node x = this.getInternalChild(tree.getRoot());
        Node older = NodeHeightMover.getOlderChild(x);
        Node younger = NodeHeightMover.getYoungerChild(x);

        for (int i = 0; i < 50; i++) {
            double logGeneticPrior = geneticPrior.calculateLogP();
            double oldHeight = x.getHeight();
            double lower = older.getHeight();
            double upper = tree.getRoot().getHeight();
            double newHeight = lower + (0.05 + 0.9 * Randomizer.nextDouble()) * (upper - lower);

            double decayRate = geneticPrior.getMutationHazardPerLength(older) + geneticPrior.getMutationHazardPerLength(younger)
                    - geneticPrior.getMutationHazardPerLength(x);
            double expectedChange = -decayRate * (newHeight - oldHeight)
                    + chain.mutations().getMutations(x).size() * Math.log((upper - newHeight) / (upper - oldHeight))
                    + chain.mutations().getMutations(older).size() * Math.log((newHeight - lower) / (oldHeight - lower))
                    + chain.mutations().getMutations(younger).size() * Math.log((newHeight - younger.getHeight()) / (oldHeight - younger.getHeight()));

            double logJacobian = NodeHeightMover.moveNode(chain.mutations(), x, newHeight, null);
            double change = geneticPrior.calculateLogP() - logGeneticPrior + logJacobian;

            assertEquals(expectedChange, change, 1e-9);
        }
    }

    /**
     * On the tree ((A,B)X,C)R with a fixed history, X may take any height between 0 and the root height 2, and the
     * mutations on the three branches around it move along with it. With a flat tree prior, the height of X has the
     * density ∝ genetic prior * Jacobian. The cells of the operator follow that density, so the chain must match the
     * mean of the density by numerical integration, and accept almost every proposal.
     */
    @Test
    public void testSamplesGeneticPriorWithMovingMutations() {
        Randomizer.setSeed(1);
        Chain chain = this.createThreeTipChain(false);
        TreeParser tree = chain.tree();
        Mutations mutations = chain.mutations();
        GeneticPrior geneticPrior = chain.geneticPrior();
        Node x = this.getInternalChild(tree.getRoot());

        MappedBatchNodeDisplacementOperator operator = new MappedBatchNodeDisplacementOperator();
        operator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "geneticPrior", geneticPrior,
                "batchSize", 1, "optimise", false, "learnTreePrior", false, "startEss", 0.0);
        operator.setOperatorSchedule(new OperatorSchedule());

        double startHeight = x.getHeight();
        List<List<Mutation>> startMutations = this.getMutations(tree, mutations);

        // the mean height under the density ∝ genetic prior * Jacobian, by integration on a grid

        int numCells = 4000;
        double logJacobian = 0.0;
        double[] logDensities = new double[numCells];
        double[] heights = new double[numCells];
        double maxLogDensity = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < numCells; i++) {
            heights[i] = 2.0 * (i + 0.5) / numCells;
            logJacobian += NodeHeightMover.moveNode(mutations, x, heights[i], operator);
            logDensities[i] = geneticPrior.calculateLogP() + logJacobian;
            maxLogDensity = Math.max(maxLogDensity, logDensities[i]);
        }
        double refMass = 0.0;
        double refMoment = 0.0;
        for (int i = 0; i < numCells; i++) {
            double density = Math.exp(logDensities[i] - maxLogDensity);
            refMass += density;
            refMoment += density * heights[i];
        }
        x.setHeight(startHeight);
        this.restoreMutations(tree, mutations, startMutations, operator);

        // run the chain with the genetic prior as the only target

        int numBurnin = 2000;
        int numSamples = 200000;
        double logTarget = geneticPrior.calculateLogP();
        double sumHeights = 0.0;
        int numAccepted = 0;

        for (int i = 0; i < numBurnin + numSamples; i++) {
            double oldHeight = x.getHeight();
            List<List<Mutation>> oldMutations = this.getMutations(tree, mutations);

            double logHastingsRatio = operator.proposal();
            double newLogTarget = geneticPrior.calculateLogP();
            double logAlpha = newLogTarget - logTarget + logHastingsRatio;

            if (Math.log(Randomizer.nextDouble()) < logAlpha) {
                logTarget = newLogTarget;
                operator.accept();
                if (i >= numBurnin) {
                    numAccepted++;
                }
            } else {
                operator.reject(logHastingsRatio == Double.NEGATIVE_INFINITY ? -2 : 0);
                x.setHeight(oldHeight);
                this.restoreMutations(tree, mutations, oldMutations, operator);
                geneticPrior.calculateLogP();
            }

            if (i >= numBurnin) {
                sumHeights += x.getHeight();
            }
        }

        assertEquals(refMoment / refMass, sumHeights / numSamples, 0.01);
        assertTrue(numAccepted > 0.97 * numSamples);
    }

    private Chain createThreeTipChain(boolean useHky) {
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", List.of(
                new Sequence("A", "GAA"), new Sequence("B", "GAC"), new Sequence("C", "CCC")
        ), "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "((A:1.0,B:1.0):1.0,C:2.0);", "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);
        mutations.initialiseMutations(this.createThreeTipHistory(tree));

        SiteModel siteModel = new SiteModel();
        if (useHky) {
            Frequencies frequencies = new Frequencies();
            frequencies.initByName("frequencies", new SimplexParam(new double[]{0.1, 0.2, 0.3, 0.4}));
            HKY hky = new HKY();
            hky.initByName("kappa", new RealScalarParam<>(3.0, PositiveReal.INSTANCE), "frequencies", frequencies);
            siteModel.initByName("substModel", hky);
        } else {
            siteModel.initByName("substModel", new beast.base.spec.evolution.substitutionmodel.JukesCantor());
        }

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName("data", alignment, "tree", tree, "siteModel", siteModel, "branchRateModel", clockModel, "mutations", mutations);
        return new Chain(tree, mutations, geneticPrior);
    }

    /**
     * Creates a history for the tree ((A,B)X,C)R with the reference GAA of the first tip: the root sequence is
     * AAA, X gains G0 at height 1.5, B gains C2 at height 0.5, and C gains C0, C1 and C2 at heights 1.5, 1.2
     * and 0.8.
     */
    private List<List<Mutation>> createThreeTipHistory(TreeParser tree) {
        Node root = tree.getRoot();
        Node x = this.getInternalChild(root);
        Node b = this.getNodeByTaxon(tree, "B");
        Node c = this.getNodeByTaxon(tree, "C");

        List<List<Mutation>> history = new ArrayList<>();
        for (int i = 0; i < tree.getNodeCount(); i++) {
            history.add(new ArrayList<>());
        }

        history.get(root.getNr()).add(new Mutation(root.getNr(), 2.0, 2.0, 0, G, A));
        history.get(x.getNr()).add(new Mutation(x.getNr(), 1.5, Mutation.BRANCH_START, 0, A, G));
        history.get(b.getNr()).add(new Mutation(b.getNr(), 0.5, Mutation.BRANCH_START, 2, A, C));
        history.get(c.getNr()).addAll(List.of(
                new Mutation(c.getNr(), 1.5, Mutation.BRANCH_START, 0, A, C),
                new Mutation(c.getNr(), 1.2, Mutation.BRANCH_START, 1, A, C),
                new Mutation(c.getNr(), 0.8, Mutation.BRANCH_START, 2, A, C)
        ));

        return history;
    }

    private List<List<Mutation>> getMutations(TreeParser tree, Mutations mutations) {
        List<List<Mutation>> branchMutations = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            branchMutations.add(mutations.getMutations(tree.getNode(nodeNr)));
        }
        return branchMutations;
    }

    private void restoreMutations(TreeParser tree, Mutations mutations, List<List<Mutation>> branchMutations, Operator operator) {
        for (Node node : tree.getNodesAsArray()) {
            mutations.applyMutations(node, branchMutations.get(node.getNr()), operator);
        }
    }

    private Node getInternalChild(Node node) {
        return node.getLeft().isLeaf() ? node.getRight() : node.getLeft();
    }

    private Node getNodeByTaxon(TreeParser tree, String taxon) {
        for (Node node : tree.getExternalNodes()) {
            if (node.getID().equals(taxon)) {
                return node;
            }
        }
        throw new IllegalArgumentException("Unknown taxon " + taxon + ".");
    }

}
