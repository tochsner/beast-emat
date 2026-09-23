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
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class GeneticPriorTest {

    /**
     * Checks the genetic prior against the closed form under Jukes-Cantor without site-rate
     * heterogeneity (supplementary Eq. 9): G = (1/4)^L * exp(-μ L T) * (μ/3)^M.
     */
    @Test
    public void testJukesCantorClosedForm() {
        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(
                        new Sequence("A", "AC"),
                        new Sequence("B", "AT"),
                        new Sequence("C", "GC"),
                        new Sequence("D", "GC")
                ),
                "dataType", "nucleotide"
        );

        // node heights: A, B, C, D at 0, (A,B) at 1, (C,D) at 0.5, root at 2

        TreeParser tree = new TreeParser();
        tree.initByName(
                "newick", "((A:1,B:1):1,(C:0.5,D:0.5):1.5);",
                "IsLabelledNewick", true,
                "adjustTipHeights", false
        );

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);

        // the reference is the sequence of A, so B needs C -> T at site 1 and (C,D) needs A -> G at site 0

        Node nodeB = this.getTip(tree, "B");
        Node nodeCD = this.getTip(tree, "C").getParent();

        List<List<Mutation>> mutationsAboveNode = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            mutationsAboveNode.add(new ArrayList<>());
        }
        mutationsAboveNode.get(nodeB.getNr()).add(new Mutation(nodeB.getNr(), 0.4, 1.0, 1, 1, 3));
        mutationsAboveNode.get(nodeCD.getNr()).add(new Mutation(nodeCD.getNr(), 1.2, 2.0, 0, 0, 2));

        mutations.initialiseMutations(mutationsAboveNode);

        // set up the genetic prior with a strict clock rate other than 1

        double clockRate = 0.5;

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", new JukesCantor());

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(clockRate, PositiveReal.INSTANCE));

        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName(
                "data", alignment,
                "tree", tree,
                "siteModel", siteModel,
                "branchRateModel", clockModel,
                "mutations", mutations
        );

        // compare with the closed form

        int numSites = 2;
        int numMutations = 2;
        double totalBranchLength = 1.0 + 1.0 + 1.0 + 0.5 + 0.5 + 1.5;

        double expectedLogP = numSites * Math.log(1.0 / 4.0)
                - clockRate * numSites * totalBranchLength
                + numMutations * Math.log(clockRate / 3.0);

        assertEquals(expectedLogP, geneticPrior.calculateLogP(), 1e-10);
    }

    /** Returns the tip with the given taxon name. */
    private Node getTip(TreeParser tree, String taxonName) {
        for (Node node : tree.getExternalNodes()) {
            if (node.getID().equals(taxonName)) {
                return node;
            }
        }
        throw new IllegalArgumentException("No tip named " + taxonName + ".");
    }

}
