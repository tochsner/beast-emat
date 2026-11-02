package emat.operators.nodetimes;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.OperatorSchedule;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

public class ClockCompensationTest {

    private static final List<Sequence> SEQUENCES = List.of(
            new Sequence("A", "ACGT"),
            new Sequence("B", "ACGA"),
            new Sequence("C", "CCGA"),
            new Sequence("D", "CTTA")
    );

    private static final String NEWICK = "(((A:0.5,B:0.5):0.5,C:0.7):0.5,D:1.3);";

    private static final int NUM_PROPOSALS = 200;

    @Test
    public void testRootRatioKeepsClockTimesTreeLength() {
        this.assertCompensates(RootRatioScaleOperator::new);
    }

    @Test
    public void testPowerRatioKeepsClockTimesTreeLength() {
        this.assertCompensates(PowerRatioScaleOperator::new);
    }

    /**
     * Runs the same proposals with and without the clock rate in the down input. The tree moves are identical, the clock rate
     * times the tree length is the same before and after, and the log Hastings ratio differs by the log Jacobian of the
     * scaling of the clock rate, which is minus the log of the factor by which the tree length changed.
     */
    private void assertCompensates(Supplier<CompensatedScaleOperator> factory) {
        Fixture plain = this.createFixture(factory, false);
        Fixture compensated = this.createFixture(factory, true);

        int numChecked = 0;
        for (int i = 0; i < NUM_PROPOSALS; i++) {
            double lengthBefore = NodeTimes.computeTreeLength(compensated.tree);
            double clockBefore = compensated.clock.get();

            Randomizer.setSeed(100 + i);
            double plainRatio = plain.operator.proposal();
            Randomizer.setSeed(100 + i);
            double compensatedRatio = compensated.operator.proposal();

            // a rejected proposal leaves the state as it was
            if (Double.isInfinite(plainRatio) || Double.isInfinite(compensatedRatio)) {
                assertEquals(plainRatio, compensatedRatio);
                continue;
            }

            double lengthAfter = NodeTimes.computeTreeLength(compensated.tree);
            assertEquals(NodeTimes.computeTreeLength(plain.tree), lengthAfter, 1e-9);
            assertEquals(clockBefore * lengthBefore, compensated.clock.get() * lengthAfter, 1e-9);
            assertEquals(plainRatio - Math.log(lengthAfter / lengthBefore), compensatedRatio, 1e-9);
            numChecked++;

            // accept, so that the next proposal starts from the new state
            plain.operator.accept();
            compensated.operator.accept();
        }

        assertNotEquals(0, numChecked);
        assertNotEquals(1.0, compensated.clock.get());
    }

    private Fixture createFixture(Supplier<CompensatedScaleOperator> factory, boolean isCompensated) {
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", SEQUENCES, "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", NEWICK, "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);

        ParsimonyMutationsInitialiser initialiser = new ParsimonyMutationsInitialiser();
        initialiser.initByName("mutations", mutations);
        initialiser.initStateNodes();

        RealScalarParam<PositiveReal> clock = new RealScalarParam<>(1.0, PositiveReal.INSTANCE);

        CompensatedScaleOperator operator = factory.get();
        if (isCompensated) {
            operator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "optimise", false, "scaleFactor", 0.1, "down", clock);
        } else {
            operator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "optimise", false, "scaleFactor", 0.1);
        }
        operator.setOperatorSchedule(new OperatorSchedule());
        return new Fixture(tree, clock, operator);
    }

    private record Fixture(TreeParser tree, RealScalarParam<PositiveReal> clock, CompensatedScaleOperator operator) {
    }

}
