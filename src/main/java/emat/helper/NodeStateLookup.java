package emat.helper;

import beast.base.evolution.tree.Node;
import emat.state.Mutations;

import java.util.function.IntUnaryOperator;

/**
 * A reusable lookup of the states at one node of an EMAT. The sequence at the node is
 * collected on the first lookup, which costs the number of mutations on the path from the
 * node up to the root, and every later lookup takes constant time. The lookup is only valid
 * while the mutations on that path stay unchanged, so it must be reset before it is used
 * for another node or after the mutations changed.
 */
public final class NodeStateLookup implements IntUnaryOperator {

    private final Mutations mutations;
    private final SiteChanges sequence;

    private Node node;
    private boolean isCollected = false;

    public NodeStateLookup(Mutations mutations, int numSites) {
        this.mutations = mutations;
        this.sequence = new SiteChanges(numSites);
    }

    /** Points the lookup to the given node, discarding the sequence collected before. */
    public NodeStateLookup reset(Node node) {
        this.node = node;
        this.isCollected = false;
        return this;
    }

    /** Returns the state of the given site at the node. */
    @Override
    public int applyAsInt(int site) {
        if (!this.isCollected) {
            MutationPaths.collectSequence(this.mutations, this.node, this.sequence);
            this.isCollected = true;
        }

        int slot = this.sequence.getSlot(site);
        return slot >= 0 ? this.sequence.getEndState(slot) : this.mutations.getReferenceSequence()[site];
    }

}
