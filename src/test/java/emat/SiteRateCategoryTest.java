package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.operators.mutations.SiteHistoryGibbsOperator;
import emat.prior.GeneticPrior;
import emat.prior.SiteRateCategories;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SiteRateCategoryTest {

    private static final double CLOCK_RATE = 0.5;

    // the sum of the branch lengths of the tree below
    private static final double TOTAL_LENGTH = 5.5;

    /** The objects of a genetic prior with three sites, two rate categories and a fixed history. */
    private record Setup(TreeParser tree, SiteModel siteModel, GeneticPrior prior, Mutations mutations) {
    }

    /**
     * Checks the genetic prior with site rate categories against the closed form under
     * Jukes-Cantor: every site contributes (1/4) exp(-μ r_s T) and every mutation at site s
     * contributes μ r_s / 3, where r_s is the rate of the category of the site.
     */
    @Test
    public void testCategoryRatesClosedForm() {
        Setup setup = createSetup(new int[]{0, 1, 0});

        double rate0 = setup.siteModel.getRateForCategory(0, setup.tree.getRoot());
        double rate1 = setup.siteModel.getRateForCategory(1, setup.tree.getRoot());

        // site 0 and site 2 are in category 0, and site 1 is in category 1, where the single mutations of sites 0 and 1 are

        double expectedLogP = 3 * Math.log(1.0 / 4.0)
                - CLOCK_RATE * TOTAL_LENGTH * (rate0 + rate1 + rate0)
                + Math.log(CLOCK_RATE * rate0 / 3.0)
                + Math.log(CLOCK_RATE * rate1 / 3.0);

        assertEquals(expectedLogP, setup.prior.calculateLogP(), 1e-10);
    }

    /**
     * Checks that the site history Gibbs operator draws the root state of every site from its
     * exact posterior under Jukes-Cantor, where the branches of a site run at the rate of its
     * category, which is fixed. On a two-tip tree with branches of length τ, the posterior is proportional to
     * π_r P_ra(r_c τ) P_rb(r_c τ) for the tip states a and b.
     */
    @Test
    public void testSiteHistoryRootStatesFollowCategories() {
        Randomizer.setSeed(5);

        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(new Sequence("A", "AA"), new Sequence("B", "CA")),
                "dataType", "nucleotide"
        );

        TreeParser tree = new TreeParser();
        tree.initByName(
                "newick", "(A:0.8,B:0.8);",
                "IsLabelledNewick", true,
                "adjustTipHeights", false
        );

        // the histories start empty, as the operator draws every history from the data

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);
        List<List<Mutation>> emptyHistories = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            emptyHistories.add(new ArrayList<>());
        }
        mutations.initialiseMutations(emptyHistories);

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", new JukesCantor(), "gammaCategoryCount", 2,
                "shape", new RealScalarParam<>(0.5, PositiveReal.INSTANCE));

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

        GeneticPrior prior = new GeneticPrior();
        prior.initByName(
                "data", alignment,
                "tree", tree,
                "siteModel", siteModel,
                "branchRateModel", clockModel,
                "mutations", mutations
        );

        // the categories are set before the first calculation, which takes them over for the operator
        SiteRateCategories categories = setCategories(prior, new int[]{0, 1});
        prior.calculateLogP();

        SiteHistoryGibbsOperator operator = new SiteHistoryGibbsOperator();
        operator.initByName("weight", 1.0, "mutations", mutations, "geneticPrior", prior,
                "siteSelection", SiteHistoryGibbsOperator.SiteSelection.UNIFORM);

        int numBurnin = 1000;
        int numSamples = 2000000;
        int[][] rootStateCounts = new int[2][4];
        for (int i = 0; i < numBurnin + numSamples; i++) {
            operator.proposal();

            if (i >= numBurnin) {
                for (int site = 0; site < 2; site++) {
                    rootStateCounts[site][getRootState(mutations, tree, site)]++;
                }
            }
        }

        // the categories are fixed, so the root state of every site follows its conditional distribution given its category

        int[][] tipStates = {{0, 1}, {0, 0}};
        for (int site = 0; site < 2; site++) {
            int category = categories.getCategory(site);
            double branchTime = 0.8 * siteModel.getRateForCategory(category, tree.getRoot());
            double[] rootProbabilities = new double[4];
            double categoryLikelihood = 0.0;
            for (int rootState = 0; rootState < 4; rootState++) {
                rootProbabilities[rootState] = 0.25
                        * getJukesCantorProbability(rootState, tipStates[site][0], branchTime)
                        * getJukesCantorProbability(rootState, tipStates[site][1], branchTime);
                categoryLikelihood += rootProbabilities[rootState];
            }

            for (int rootState = 0; rootState < 4; rootState++) {
                assertEquals(rootProbabilities[rootState] / categoryLikelihood, (double) rootStateCounts[site][rootState] / numSamples, 0.005,
                        "site " + site + ", root state " + rootState + ".");
            }
        }
    }

    /** Returns the Jukes-Cantor transition probability from one state to another over the given time. */
    private static double getJukesCantorProbability(int from, int to, double time) {
        double decay = Math.exp(-4.0 * time / 3.0);
        return from == to ? 0.25 + 0.75 * decay : 0.25 - 0.25 * decay;
    }

    /** Returns the state of the given site at the root, which is the reference state changed by the mutations above the root. */
    private static int getRootState(Mutations mutations, TreeParser tree, int site) {
        int state = mutations.getReferenceSequence()[site];
        for (Mutation mutation : mutations.getMutations(tree.getRoot())) {
            if (mutation.site() == site) {
                state = mutation.newState();
            }
        }
        return state;
    }

    /**
     * Creates a genetic prior on a four-tip tree with three sites, of which the first two carry
     * one mutation each and the third carries none, with the given categories of the sites.
     */
    private static Setup createSetup(int[] categoryValues) {
        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(
                        new Sequence("A", "ACA"),
                        new Sequence("B", "ATA"),
                        new Sequence("C", "GCA"),
                        new Sequence("D", "GCA")
                ),
                "dataType", "nucleotide"
        );

        TreeParser tree = new TreeParser();
        tree.initByName(
                "newick", "((A:1,B:1):1,(C:0.5,D:0.5):1.5);",
                "IsLabelledNewick", true,
                "adjustTipHeights", false
        );

        // the reference is the sequence of A, so B needs C -> T at site 1 and (C,D) needs A -> G at site 0

        Node nodeB = getTip(tree, "B");
        Node nodeCD = getTip(tree, "C").getParent();

        List<List<Mutation>> mutationsAboveNode = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            mutationsAboveNode.add(new ArrayList<>());
        }
        mutationsAboveNode.get(nodeB.getNr()).add(new Mutation(nodeB.getNr(), 0.4, Mutation.BRANCH_START, 1, 1, 3));
        mutationsAboveNode.get(nodeCD.getNr()).add(new Mutation(nodeCD.getNr(), 1.2, Mutation.BRANCH_START, 0, 0, 2));

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);
        mutations.initialiseMutations(mutationsAboveNode);

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", new JukesCantor(), "gammaCategoryCount", 2,
                "shape", new RealScalarParam<>(0.5, PositiveReal.INSTANCE));

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(CLOCK_RATE, PositiveReal.INSTANCE));

        GeneticPrior prior = new GeneticPrior();
        prior.initByName(
                "data", alignment,
                "tree", tree,
                "siteModel", siteModel,
                "branchRateModel", clockModel,
                "mutations", mutations
        );
        setCategories(prior, categoryValues);

        return new Setup(tree, siteModel, prior, mutations);
    }

    /**
     * Sets the categories of the sites of the given prior and marks them as redrawn, so that the
     * next calculation of the prior takes them over. Returns the categories of the prior.
     */
    private static SiteRateCategories setCategories(GeneticPrior prior, int[] categoryValues) {
        SiteRateCategories categories = prior.getSiteRateCategories();
        for (int site = 0; site < categoryValues.length; site++) {
            categories.setStateCategory(site, categoryValues[site]);
        }
        categories.markRedrawn();
        return categories;
    }

    /** Returns the tip with the given taxon name. */
    private static Node getTip(TreeParser tree, String taxonName) {
        for (Node node : tree.getExternalNodes()) {
            if (node.getID().equals(taxonName)) {
                return node;
            }
        }
        throw new IllegalArgumentException("No tip named " + taxonName + ".");
    }

}
