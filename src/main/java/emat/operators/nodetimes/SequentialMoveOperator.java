package emat.operators.nodetimes;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

@Description("Proposes the move of one operator and then the move of a second operator from the state the first one " +
        "produced, as a single step that is accepted or rejected as a whole. The reverse of this move undoes the second move " +
        "and then the first, so the log Hastings ratio is the sum of the two. The two operators are not tuned by this " +
        "operator: they keep their own entries in the operator schedule, which tune them on their own proposals.")
public class SequentialMoveOperator extends Operator {

    final public Input<Operator> firstInput = new Input<>("first", "the operator that moves first", Input.Validate.REQUIRED);
    final public Input<Operator> secondInput = new Input<>("second", "the operator that moves second, from the state after the first move", Input.Validate.REQUIRED);

    private Operator first;
    private Operator second;

    @Override
    public void initAndValidate() {
        this.first = this.firstInput.get();
        this.second = this.secondInput.get();
    }

    @Override
    public double proposal() {
        double firstLogHastingsRatio = this.first.proposal();
        if (firstLogHastingsRatio == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        double secondLogHastingsRatio = this.second.proposal();
        if (secondLogHastingsRatio == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        return firstLogHastingsRatio + secondLogHastingsRatio;
    }

    @Override
    public List<StateNode> listStateNodes() {
        Set<StateNode> stateNodes = new LinkedHashSet<>(this.first.listStateNodes());
        stateNodes.addAll(this.second.listStateNodes());
        return new ArrayList<>(stateNodes);
    }

    @Override
    public boolean requiresStateInitialisation() {
        return this.first.requiresStateInitialisation() || this.second.requiresStateInitialisation();
    }

}
