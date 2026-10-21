package emat;

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
import emat.helper.SiteMutations;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.MissingTipGibbsOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.StochasticMapping;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MissingTipGibbsOperatorTest {

    /**
     * On the tree ((A:0.3,B:0.5):0.4,C:0.9) under HKY, tip B is missing at site 0 and
     * observed at site 1. As only this operator runs, the state x of the parent of B stays
     * fixed, so the state of B at site 0 must follow P_x·(0.5) and the branch must be free of
     * mutations there with probability exp(-q_x 0.5). The history at site 1 must stay as it is.
     */
    @Test
    public void testMissingSiteFollowsForwardProcess() {
        Randomizer.setSeed(1);

        TestSetup setup = new TestSetup(
                List.of(new Sequence("A", "AC"), new Sequence("B", "NA"), new Sequence("C", "GC")),
                "((A:0.3,B:0.5):0.4,C:0.9);"
        );

        Node tip = setup.getNode("B");
        int parentState = MutationPaths.getState(setup.mutations, tip.getParent(), 0);
        List<Mutation> observedMutations = SiteMutations.collect(setup.mutations, tip, 1);

        // run the operator and collect the end state and whether the branch has mutations

        int numSamples = 400000;
        double[] stateCounts = new double[4];
        double numWithoutMutations = 0;

        for (int i = 0; i < numSamples; i++) {
            assertEquals(Double.POSITIVE_INFINITY, setup.operator.proposal());
            setup.mutations.setEverythingDirty(false);

            stateCounts[MutationPaths.getState(setup.mutations, tip, 0)]++;
            if (SiteMutations.collect(setup.mutations, tip, 0).isEmpty()) {
                numWithoutMutations++;
            }
        }

        assertEquals(observedMutations, SiteMutations.collect(setup.mutations, tip, 1), "The observed site changed.");

        // compare with the forward process from the parent state

        double[] transitionProbabilities = StochasticMapping.computeTransitionProbabilities(setup.substitutionModel.getEigenDecomposition(null), 0.5);
        for (int state = 0; state < 4; state++) {
            assertEquals(transitionProbabilities[parentState * 4 + state], stateCounts[state] / numSamples, 0.005,
                    "Parent state " + parentState + ", tip state " + state + ".");
        }

        double escapeRate = -setup.geneticPrior.computeRateMatrix()[parentState * 4 + parentState];
        assertEquals(Math.exp(-escapeRate * 0.5), numWithoutMutations / numSamples, 0.005);
    }

    @Test
    public void testRejectsWithoutMissingTips() {
        TestSetup setup = new TestSetup(
                List.of(new Sequence("A", "AC"), new Sequence("B", "RA"), new Sequence("C", "GC")),
                "((A:0.3,B:0.5):0.4,C:0.9);"
        );

        assertEquals(Double.NEGATIVE_INFINITY, setup.operator.proposal());
    }

    /* Test Setup */

    /**
     * Builds the tree, alignment, mutations (initialised by parsimony), a genetic prior under
     * HKY with clock rate 1, and the operator.
     */
    private static class TestSetup {

        final TreeParser tree;
        final Mutations mutations;
        final SubstitutionModel substitutionModel;
        final GeneticPrior geneticPrior;
        final MissingTipGibbsOperator operator;

        TestSetup(List<Sequence> sequences, String newick) {
            Alignment alignment = new Alignment();
            alignment.initByName("sequence", sequences, "dataType", "nucleotide");

            this.tree = new TreeParser();
            this.tree.initByName("newick", newick, "IsLabelledNewick", true, "adjustTipHeights", false);

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

            this.geneticPrior = new GeneticPrior();
            this.geneticPrior.initByName(
                    "data", alignment,
                    "tree", this.tree,
                    "siteModel", siteModel,
                    "branchRateModel", clockModel,
                    "mutations", this.mutations
            );

            this.operator = new MissingTipGibbsOperator();
            this.operator.initByName("weight", 1.0, "mutations", this.mutations, "geneticPrior", this.geneticPrior);
        }

        Node getNode(String taxon) {
            for (Node node : this.tree.getExternalNodes()) {
                if (node.getID().equals(taxon)) {
                    return node;
                }
            }
            throw new IllegalArgumentException("Unknown taxon " + taxon);
        }

    }

}
