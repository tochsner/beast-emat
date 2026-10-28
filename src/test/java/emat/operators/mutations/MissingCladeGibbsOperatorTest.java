package emat.operators.mutations;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.Frequencies;
import beast.base.spec.evolution.substitutionmodel.HKY;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.SimplexParam;
import beast.base.util.Randomizer;
import emat.helper.MutationPaths;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.prior.GeneticPrior;
import emat.state.Mutations;
import emat.stochasticmapping.SubstitutionProcess;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MissingCladeGibbsOperatorTest {

    // on ((A,B)X,C)R, site 0 is missing at A and B, so the clade X is missing; site 1 is missing at A only
    private static final List<Sequence> SEQUENCES = List.of(new Sequence("A", "NN"), new Sequence("B", "NC"), new Sequence("C", "GC"));
    private static final String NEWICK = "((A:0.3,B:0.5)X:0.4,C:0.9)R;";

    private static final int C = 1;
    private static final int G = 2;

    /**
     * As only this operator runs, at site 0 the clade X is missing and attaches to the root,
     * whose other child C is observed in G, so the root state follows π_r·P_rG(0.9), and X, A
     * and B follow from it by P_r·(0.4), P_r·(0.7) and P_r·(0.9). At site 1, the clade stops at
     * A, whose sibling B is observed in C, so the root state r stays fixed, the state of X
     * follows P_rx(0.4)·P_xC(0.5), and A follows from it by P_x·(0.3).
     */
    @Test
    public void testMissingCladeAndAttachmentFollowPosterior() {
        Randomizer.setSeed(1);
        TestSetup setup = new TestSetup(Integer.MAX_VALUE);

        Node root = setup.tree.getRoot();
        Node tipA = setup.getNode("A");
        Node tipB = setup.getNode("B");
        Node inner = tipA.getParent();
        int rootStateAtSite1 = MutationPaths.getState(setup.mutations, root, 1);

        // run the operator and collect the states at both sites

        int numSamples = 400000;
        double[][] stateCounts = new double[6][4];
        for (int i = 0; i < numSamples; i++) {
            assertEquals(Double.POSITIVE_INFINITY, setup.operator.proposal());
            setup.mutations.setEverythingDirty(false);

            stateCounts[0][MutationPaths.getState(setup.mutations, root, 0)]++;
            stateCounts[1][MutationPaths.getState(setup.mutations, inner, 0)]++;
            stateCounts[2][MutationPaths.getState(setup.mutations, tipA, 0)]++;
            stateCounts[3][MutationPaths.getState(setup.mutations, tipB, 0)]++;
            stateCounts[4][MutationPaths.getState(setup.mutations, inner, 1)]++;
            stateCounts[5][MutationPaths.getState(setup.mutations, tipA, 1)]++;
        }

        assertEquals(rootStateAtSite1, MutationPaths.getState(setup.mutations, root, 1), "The root state at site 1 changed.");
        assertEquals(C, MutationPaths.getState(setup.mutations, tipB, 1), "The observed tip B changed at site 1.");

        // compare with the posterior: the attachment node from its star, and the clade forward from it

        double[] frequencies = setup.substitutionModel.getFrequencies();
        double[] rootPosterior = new double[4];
        double[] innerPosterior = new double[4];
        for (int state = 0; state < 4; state++) {
            rootPosterior[state] = frequencies[state] * setup.getTransitionProbabilities(0.9)[state * 4 + G];
            innerPosterior[state] = setup.getTransitionProbabilities(0.4)[rootStateAtSite1 * 4 + state] * setup.getTransitionProbabilities(0.5)[state * 4 + C];
        }
        normalise(rootPosterior);
        normalise(innerPosterior);

        double[][] expected = {
                rootPosterior,
                propagate(rootPosterior, setup.getTransitionProbabilities(0.4)),
                propagate(rootPosterior, setup.getTransitionProbabilities(0.7)),
                propagate(rootPosterior, setup.getTransitionProbabilities(0.9)),
                innerPosterior,
                propagate(innerPosterior, setup.getTransitionProbabilities(0.3)),
        };
        String[] names = {"R at site 0", "X at site 0", "A at site 0", "B at site 0", "X at site 1", "A at site 1"};

        for (int k = 0; k < expected.length; k++) {
            for (int state = 0; state < 4; state++) {
                assertEquals(expected[k][state], stateCounts[k][state] / numSamples, 0.005, names[k] + ", state " + state + ".");
            }
        }
    }

    private static void normalise(double[] weights) {
        double total = 0.0;
        for (double weight : weights) {
            total += weight;
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] /= total;
        }
    }

    /** Returns the distribution of the state at the end of a branch with the given transition probabilities, given the distribution at its start. */
    private static double[] propagate(double[] startDistribution, double[] transitionProbabilities) {
        double[] endDistribution = new double[4];
        for (int start = 0; start < 4; start++) {
            for (int end = 0; end < 4; end++) {
                endDistribution[end] += startDistribution[start] * transitionProbabilities[start * 4 + end];
            }
        }
        return endDistribution;
    }

    /**
     * Without any levels, the clades are the branches above the picked tips, which attach to X,
     * so X is redrawn while the root never changes.
     */
    @Test
    public void testWithoutLevelsKeepsRootStates() {
        Randomizer.setSeed(2);
        TestSetup setup = new TestSetup(0);

        Node root = setup.tree.getRoot();
        Node inner = setup.getNode("A").getParent();
        int rootState = MutationPaths.getState(setup.mutations, root, 0);
        int numChangedInnerStates = 0;

        for (int i = 0; i < 20000; i++) {
            int oldInnerState = MutationPaths.getState(setup.mutations, inner, 0);
            setup.operator.proposal();
            setup.mutations.setEverythingDirty(false);

            assertEquals(rootState, MutationPaths.getState(setup.mutations, root, 0));
            if (MutationPaths.getState(setup.mutations, inner, 0) != oldInnerState) {
                numChangedInnerStates++;
            }
        }

        assertEquals(true, numChangedInnerStates > 0, "The operator never changed the state of the attachment node X.");
    }

    /* Test Setup */

    /**
     * Builds the tree, alignment, mutations (initialised by parsimony), a genetic prior under
     * HKY with clock rate 1, and the operator with the given maximum number of levels.
     */
    private static class TestSetup {

        final TreeParser tree;
        final Mutations mutations;
        final SubstitutionModel substitutionModel;
        final MissingCladeGibbsOperator operator;

        TestSetup(int maxLevels) {
            Alignment alignment = new Alignment();
            alignment.initByName("sequence", SEQUENCES, "dataType", "nucleotide");

            this.tree = new TreeParser();
            this.tree.initByName("newick", NEWICK, "IsLabelledNewick", true, "adjustTipHeights", false);

            this.mutations = new Mutations();
            this.mutations.initByName("tree", this.tree, "alignment", alignment);

            ParsimonyMutationsInitialiser initialiser = new ParsimonyMutationsInitialiser();
            initialiser.initByName("mutations", this.mutations);
            initialiser.initStateNodes();

            Frequencies frequencies = new Frequencies();
            frequencies.initByName("frequencies", new SimplexParam(new double[]{0.1, 0.2, 0.3, 0.4}));
            HKY hky = new HKY();
            hky.initByName("kappa", new RealScalarParam<>(4.0, PositiveReal.INSTANCE), "frequencies", frequencies);
            this.substitutionModel = hky;

            SiteModel siteModel = new SiteModel();
            siteModel.initByName("substModel", hky);

            StrictClockModel clockModel = new StrictClockModel();
            clockModel.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

            GeneticPrior geneticPrior = new GeneticPrior();
            geneticPrior.initByName(
                    "data", alignment,
                    "tree", this.tree,
                    "siteModel", siteModel,
                    "branchRateModel", clockModel,
                    "mutations", this.mutations
            );

            this.operator = new MissingCladeGibbsOperator();
            this.operator.initByName("weight", 1.0, "mutations", this.mutations, "geneticPrior", geneticPrior, "maxLevels", maxLevels);
        }

        Node getNode(String taxon) {
            for (Node node : this.tree.getExternalNodes()) {
                if (node.getID().equals(taxon)) {
                    return node;
                }
            }
            throw new IllegalArgumentException("Unknown taxon " + taxon);
        }

        double[] getTransitionProbabilities(double time) {
            return SubstitutionProcess.computeTransitionProbabilities(this.substitutionModel.getEigenDecomposition(null), time);
        }

    }

}
