package esat.adapters;

import adapters.Adapter;
import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.StateNode;
import esat.state.NodeStates;

import java.util.List;

@Description("Immutable adapter that summarises the node states by the number of differences between the states of every node and of its parent.")
public class NodeStatesCountAdapter extends BEASTObject implements Adapter {

    public final Input<NodeStates> nodeStatesInput = new Input<>("nodeStates", "Node states to count the differences of.", Input.Validate.REQUIRED);

    private NodeStates nodeStates;

    @Override
    public void initAndValidate() {
        this.nodeStates = this.nodeStatesInput.get();
    }

    @Override
    public int getNumImmutable() {
        return 1;
    }

    @Override
    public int getNumMutable() {
        return 0;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        return new double[]{this.nodeStates.getDifferenceCount()};
    }

    @Override
    public double[] getMutable(int nodeId) {
        return new double[0];
    }

    @Override
    public double update(double[] mutable, int nodeId) {
        return 0.0;
    }

    @Override
    public double getLogJacobianCorrection(int nodeId) {
        return 0.0;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.nodeStates);
    }
}
