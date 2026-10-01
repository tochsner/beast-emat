package emat.helper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A reusable map from sites to their states at two points of the tree, the start and the
 * end. Entries are kept in slots in the order they were added, and a lookup table over all
 * sites finds the slot of a site in constant time. Clearing only resets the
 * added sites, so its cost does not depend on the genome length.
 */
public final class SiteChanges {

    private record Entry(int site, int startState, int endState) {}

    // the slot of every site, or -1 if the site has no entry
    private final int[] slotOfSite;
    private final List<Entry> entries = new ArrayList<>();

    public SiteChanges(int numSites) {
        this.slotOfSite = new int[numSites];
        Arrays.fill(this.slotOfSite, -1);
    }

    /** Returns the number of sites with an entry, which are stored in the slots below it. */
    public int getSize() {
        return this.entries.size();
    }

    public int getSite(int slot) {
        return this.entries.get(slot).site();
    }

    public int getStartState(int slot) {
        return this.entries.get(slot).startState();
    }

    public int getEndState(int slot) {
        return this.entries.get(slot).endState();
    }

    /** Returns the slot of the given site, or -1 if it has no entry. */
    public int getSlot(int site) {
        return this.slotOfSite[site];
    }

    public boolean containsSite(int site) {
        return this.slotOfSite[site] >= 0;
    }

    /** Adds an entry for the given site, which must not have one yet, and returns its slot. */
    public int addSite(int site, int startState, int endState) {
        int slot = this.entries.size();
        this.entries.add(new Entry(site, startState, endState));
        this.slotOfSite[site] = slot;
        return slot;
    }

    public void setStartState(int slot, int state) {
        Entry entry = this.entries.get(slot);
        this.entries.set(slot, new Entry(entry.site(), state, entry.endState()));
    }

    /** Removes all entries. */
    public void clear() {
        for (Entry entry : this.entries) {
            this.slotOfSite[entry.site()] = -1;
        }
        this.entries.clear();
    }

}
