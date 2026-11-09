package emat.prior;

/**
 * The rate categories of the sites of a genetic prior. Operators change the categories of the state
 * as a whole, and then mark them as redrawn. The genetic prior reads a snapshot of the categories,
 * which it updates in every calculation. The snapshot is stored with the branches, so that on restore
 * the categories match the cached branches again. BEAST stores the calculation nodes only after the
 * proposal, when the state may already be redrawn, so the state is not stored itself. After every
 * accepted or restored calculation the state equals the snapshot, so a restore copies the restored
 * snapshot back into the state. The generation counts the changes of the snapshot, so that its
 * readers can tell whether it changed.
 */
public class SiteRateCategories {

    // the categories of the state, which operators change
    private final int[] stateCategories;

    // the categories of the snapshot, which the cached branches were computed with
    private int[] siteCategories;
    private int[] storedSiteCategories;

    // whether the state was marked as redrawn since the last update, and since it was stored
    private boolean isChanged = false;
    private boolean isChangedSinceStore = false;

    private long generation = 0;

    public SiteRateCategories(int numSites) {
        this.stateCategories = new int[numSites];
        this.siteCategories = new int[numSites];
        this.storedSiteCategories = new int[numSites];
    }

    /**
     * Copies the categories of the state into the snapshot if they were marked as redrawn since the
     * last update, and returns whether they were. The categories are redrawn as a whole, so the
     * snapshot is copied as a whole.
     */
    public boolean update() {
        if (!this.isChanged) {
            return false;
        }
        this.isChanged = false;

        System.arraycopy(this.stateCategories, 0, this.siteCategories, 0, this.siteCategories.length);
        this.generation++;
        return true;
    }

    /** Returns the category of the given site in the snapshot, as of the last update. */
    public int getCategory(int site) {
        return this.siteCategories[site];
    }

    /** Returns the category of the given site in the state, which may differ from the snapshot. */
    public int getStateCategory(int site) {
        return this.stateCategories[site];
    }

    /** Sets the category of the given site in the state. The categories have to be marked as redrawn afterwards. */
    public void setStateCategory(int site, int category) {
        this.stateCategories[site] = category;
    }

    /** Marks the categories of the state as redrawn, so that the next update copies them into the snapshot. */
    public void markRedrawn() {
        this.isChanged = true;
        this.isChangedSinceStore = true;
    }

    /** Returns the generation of the snapshot, which changes whenever the snapshot may have changed. */
    public long getGeneration() {
        return this.generation;
    }

    /** Stores the snapshot, so that it can be restored with the branches. */
    public void store() {
        System.arraycopy(this.siteCategories, 0, this.storedSiteCategories, 0, this.siteCategories.length);
        this.isChangedSinceStore = false;
    }

    /**
     * Restores the stored snapshot, and the state from it. The generation changes if the state was
     * redrawn since it was stored, so that the readers of the snapshot compare it again.
     */
    public void restore() {
        int[] siteCategories = this.siteCategories;
        this.siteCategories = this.storedSiteCategories;
        this.storedSiteCategories = siteCategories;

        System.arraycopy(this.siteCategories, 0, this.stateCategories, 0, this.stateCategories.length);
        this.isChanged = false;

        if (this.isChangedSinceStore) {
            this.generation++;
            this.isChangedSinceStore = false;
        }
    }

}
