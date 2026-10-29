package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.Taxon;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.Frequencies;
import beast.base.spec.evolution.substitutionmodel.HKY;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.SimplexParam;
import beast.base.util.Randomizer;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.nodetimes.IntervalScaleOperator;
import emat.operators.topology.MutationDirectedSprOperator;
import emat.operators.mutations.MutationTimeOperator;
import emat.operators.mutations.SiteHistoryGibbsOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutations;
import emat.tree.IncrementalStoreTree;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class GeneticPriorCachingTest {

    private static final String NEWICK = "(((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2):0.3,(E:0.6,F:0.6):0.4);";

    /**
     * Runs a Metropolis-Hastings chain with every operator, driving the state through the
     * same store, accept and restore cycle as EmatMCMC, and checks after every step that
     * the incrementally updated genetic prior matches a computation from scratch.
     */
    @Test
    public void testIncrementalMatchesFullComputation() {
        TreeParser tree = new TreeParser();
        tree.initByName(
                "newick", NEWICK,
                "IsLabelledNewick", true,
                "adjustTipHeights", false
        );
        this.runChain(tree);
    }

    /**
     * Runs the same chain on an incremental store tree, for which the genetic prior only
     * visits the nodes flagged by the tree and the branches marked dirty by the mutations.
     */
    @Test
    public void testIncrementalMatchesFullComputationOnIncrementalStoreTree() {
        TreeParser parsedTree = new TreeParser();
        parsedTree.initByName(
                "newick", NEWICK,
                "IsLabelledNewick", true,
                "adjustTipHeights", false
        );

        List<Taxon> taxa = parsedTree.getTaxonset().asStringList().stream().map(Taxon::new).toList();
        IncrementalStoreTree tree = new IncrementalStoreTree();
        tree.initByName("taxonset", new TaxonSet(taxa));
        tree.assignFrom(parsedTree);
        this.runChain(tree);
    }

    private void runChain(Tree tree) {
        Randomizer.setSeed(3);

        // set up the alignment and the mutations

        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(
                        new Sequence("A", "ACGTACGTAC"),
                        new Sequence("B", "ACNAACGNNC"),
                        new Sequence("C", "TCGAACCTAC"),
                        new Sequence("D", "TCGTACCTAG"),
                        new Sequence("E", "GCGTTNNNNN"),
                        new Sequence("F", "GCGTTCGAAC")
                ),
                "dataType", "nucleotide"
        );

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);

        ParsimonyMutationsInitialiser initialiser = new ParsimonyMutationsInitialiser();
        initialiser.initByName("mutations", mutations);
        initialiser.initStateNodes();

        SiteModel siteModel = new SiteModel();
        Frequencies frequencies = new Frequencies();
        frequencies.initByName("frequencies", new SimplexParam(new double[]{0.1, 0.2, 0.3, 0.4}));

        // HKY has state-dependent escape rates, so the total mutation rate changes with the mutations
        HKY hky = new HKY();
        hky.initByName("kappa", new RealScalarParam<>(3.0, PositiveReal.INSTANCE), "frequencies", frequencies);

        siteModel.initByName("substModel", hky);

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(0.5, PositiveReal.INSTANCE));

        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName(
                "data", alignment,
                "tree", tree,
                "siteModel", siteModel,
                "branchRateModel", clockModel,
                "mutations", mutations
        );

        // set up the operators

        MutationDirectedSprOperator mdSprOperator = new MutationDirectedSprOperator();
        mdSprOperator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "geneticPrior", geneticPrior);

        MutationTimeOperator mutationTimeOperator = new MutationTimeOperator();
        mutationTimeOperator.initByName("weight", 1.0, "mutations", mutations);

        IntervalScaleOperator intervalScaleOperator = new IntervalScaleOperator();
        intervalScaleOperator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "scaleFactor", 0.5, "optimise", false);

        SiteHistoryGibbsOperator gibbsOperator = new SiteHistoryGibbsOperator();
        gibbsOperator.initByName("weight", 1.0, "mutations", mutations, "geneticPrior", geneticPrior);

        List<Operator> operators = List.of(
                mdSprOperator, mutationTimeOperator, intervalScaleOperator, gibbsOperator
        );

        // set up the state with the genetic prior as the posterior

        State state = new State();
        state.initByName("stateNode", tree, "stateNode", mutations);
        state.initialise();
        state.setPosterior(geneticPrior);
        state.setEverythingDirty(true);

        double oldLogP = geneticPrior.calculateLogP();
        state.setEverythingDirty(false);

        // run the chain like EmatMCMC

        for (long i = 0; i < 20000; i++) {
            state.store(i);
            Operator operator = operators.get(Randomizer.nextInt(operators.size()));

            double logHastingsRatio = operator.proposal();
            if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
                state.restore();
                assertEquals(this.computeLogPFromScratch(geneticPrior), oldLogP, 1e-9);
                continue;
            }

            state.storeCalculationNodes();
            state.checkCalculationNodesDirtiness();

            double newLogP = geneticPrior.calculateLogP();
            assertEquals(this.computeLogPFromScratch(geneticPrior), newLogP, 1e-9, "Proposal of " + operator.getClass().getSimpleName());

            if (Math.log(Randomizer.nextDouble()) < newLogP - oldLogP + logHastingsRatio) {
                oldLogP = newLogP;
                state.acceptCalculationNodes();
            } else {
                state.restore();
                state.restoreCalculationNodes();
                assertEquals(this.computeLogPFromScratch(geneticPrior), geneticPrior.getCurrentLogP(), 1e-9, "Rejection of " + operator.getClass().getSimpleName());
            }

            state.setEverythingDirty(false);
        }
    }

    /**
     * Computes the genetic prior of the current tree and mutations from scratch with a new
     * genetic prior on the same inputs, whose first calculation recomputes every branch.
     */
    private double computeLogPFromScratch(GeneticPrior geneticPrior) {
        GeneticPrior freshGeneticPrior = new GeneticPrior();
        freshGeneticPrior.initByName(
                "data", geneticPrior.dataInput.get(),
                "tree", geneticPrior.treeInput.get(),
                "siteModel", geneticPrior.siteModelInput.get(),
                "branchRateModel", geneticPrior.branchRateModelInput.get(),
                "mutations", geneticPrior.mutationsInput.get()
        );
        return freshGeneticPrior.calculateLogP();
    }

}
