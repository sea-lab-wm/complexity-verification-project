import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;

public class MyExpenses {

    public static final int TYPE_TRANSACTION = 0;
    public static final int TYPE_TRANSFER = 1;
    public static final int TYPE_SPLIT = 2;
    public static final long TRESHOLD_REMIND_CONTRIB = 113L;
    private static final String KEY_ACCOUNTID = "";
    private static final Object TASK_PRINT = null;
    private static final Dummy KEY_ROWID = null;
    private StickyListHeadersAdapter mDrawerListAdapter;
    private long mAccountId = 0;
    private Dummy mList;
    private Dummy payeeToId;
    private Font currentFont;
    private boolean mAreHeadersSticky;
    private Dummy mAdapter;
    private Dummy[] files;

    public enum ContribFeature {
        DISTRIBUTION,
        SPLIT_TRANSACTION,
        PRINT
    }

    //ADDED BY US
    public void runAll() {
        contribFeatureCalled(ContribFeature.DISTRIBUTION, new Serializable() {});
        getContentProviderOperationsForCreate(new TransactionChange(), 1, 1);
        mergeUpdate(new TransactionChange(), new TransactionChange());
        try {
            processChar(new char[5], 1, new StringBuffer());
        } catch (MyExpenses.DocumentException | IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        updateOrClearHeader(1);
    }

    // org.totschnig.myexpenses.activity.MyExpenses.contribFeatureCalled(org.totschnig.myexpenses.model.ContribFeature,java.io.Serializable)
//  @Override // removed to allow compilation
    //SNIPPET_STARTS
    public void contribFeatureCalled(ContribFeature feature, Serializable tag) {
        switch (feature) {
            case DISTRIBUTION:
                Account a = Account.getInstanceFromDb(mAccountId);
                recordUsage(feature);
                Intent i = new Intent(this, ManageCategories.class);
                i.setAction("myexpenses.intent.distribution");
                i.putExtra(KEY_ACCOUNTID, mAccountId);
                if (tag != null) {
                    int year = (int) ((Long) tag / 1000);
                    int groupingSecond = (int) ((Long) tag % 1000);
                    i.putExtra("grouping", a != null ? a.grouping : Grouping.NONE);
                    i.putExtra("groupingYear", year);
                    i.putExtra("groupingSecond", groupingSecond);
                }
                startActivity(i);
                break;
            case SPLIT_TRANSACTION:
                if (tag != null) {
                    startTaskExecution(
                            TaskExecutionFragment.TASK_SPLIT,
                            (Object[]) tag,
                            null,
                            0);
                }
                break;
            case PRINT:
                TransactionList tl = getCurrentFragment();
                if (tl != null) {
                    Bundle args = new Bundle();
                    args.putSparseParcelableArray(TransactionList.KEY_FILTER, tl.getFilterCriteria());
                    args.putLong(KEY_ROWID, mAccountId);
                    getSupportFragmentManager().beginTransaction()
                            .add(TaskExecutionFragment.newInstanceWithBundle(args, TASK_PRINT),
                                    ProtectionDelegate.ASYNC_TAG)
                            .add(ProgressDialogFragment.newInstance(R.string.progress_dialog_printing), ProtectionDelegate.PROGRESS_TAG)
                            .commit();
                }
                break;
        }
    }

    // org.totschnig.myexpenses.sync.SyncAdapter.getContentProviderOperationsForCreate(org.totschnig.myexpenses.sync.json.TransactionChange,int,int)
    //SNIPPET_STARTS
private ArrayList<ContentProviderOperation> getContentProviderOperationsForCreate(
        TransactionChange change, int offset, int parentOffset) {
    if (!change.isCreate()) throw new AssertionError();
    Long amount;
    if (change.amount() != null) {
        amount = change.amount();
    } else {
        amount = 0L;
    }
    Transaction t;
    long transferAccount;
    if (change.splitParts() != null) {
        t = new SplitTransaction(getAccount().getId(), amount);
    } else if (change.transferAccount() != null &&
            (transferAccount = extractTransferAccount(change.transferAccount(), change.label())) != -1) {
        t = new Transfer(getAccount().getId(), amount);
        t.transfer_account = transferAccount;
    } else {
        t = new Transaction(getAccount().getId(), amount);
        if (change.label() != null) {
            long catId = extractCatId(change.label());
            if (catId != -1) {
                t.setCatId(catId);
            }
        }
    }
    t.uuid = change.uuid();
    if (change.comment() != null) {
        t.comment = change.comment();
    }
    if (change.date() != null) {
        Long date = change.date();
        assert date != null;
        t.setDate(new Date(date * 1000));
    }

    if (change.payeeName() != null) {
        long id = Payee.extractPayeeId(change.payeeName(), payeeToId);
        if (id != -1) {
            t.payeeId = id;
        }
    }
    if (change.methodLabel() != null) {
        long id = extractMethodId(change.methodLabel());
        if (id != -1) {
            t.methodId = id;
        }
    }
    if (change.crStatus() != null) {
        t.crStatus = Transaction.CrStatus.valueOf(change.crStatus());
    }
    t.referenceNumber = change.referenceNumber();
    if (parentOffset == -1 && change.parentUuid() != null) {
        long parentId = Transaction.findByUuid(change.parentUuid());
        if (parentId == -1) {
            return new ArrayList<>(); //if we fail to link a split part to a parent, we need to ignore it
        }
        t.parentId = parentId;
    }
    if (change.pictureUri() != null) {
        t.setPictureUri(Uri.parse(change.pictureUri()));
    }
    return t.buildSaveOperations(offset, parentOffset, true);
}

    private long extractMethodId(Dummy methodLabel) {
        return 0;
    }

    private long extractTransferAccount(Dummy transferAccount, Dummy label) {
        return 0;
    }

    private long extractCatId(Dummy label) {
        return 0;
    }

    private Dummy getAccount() {
        return null;
    }

    // org.totschnig.myexpenses.sync.SyncAdapter.mergeUpdate(org.totschnig.myexpenses.sync.json.TransactionChange,org.totschnig.myexpenses.sync.json.TransactionChange)
    //SNIPPET_STARTS
    private TransactionChange mergeUpdate(TransactionChange initial, TransactionChange change) {
        if (!(change.isCreateOrUpdate() && initial.isCreateOrUpdate())) {
            throw new IllegalStateException("Can only merge creates and updates");
        }
        if (!initial.uuid().equals(change.uuid())) {
            throw new IllegalStateException("Can only merge changes with same uuid");
        }
        TransactionChange.Builder builder = initial.toBuilder();
        if (change.parentUuid() != null) {
            builder.setParentUuid(change.parentUuid());
        }
        if (change.comment() != null) {
            builder.setComment(change.comment());
        }
        if (change.date() != null) {
            builder.setDate(change.date());
        }
        if (change.amount() != null) {
            builder.setAmount(change.amount());
        }
        if (change.label() != null) {
            builder.setLabel(change.label());
        }
        if (change.payeeName() != null) {
            builder.setPayeeName(change.payeeName());
        }
        if (change.transferAccount() != null) {
            builder.setTransferAccount(change.transferAccount());
        }
        if (change.methodLabel() != null) {
            builder.setMethodLabel(change.methodLabel());
        }
        if (change.crStatus() != null) {
            builder.setCrStatus(change.crStatus());
        }
        if (change.referenceNumber() != null) {
            builder.setReferenceNumber(change.referenceNumber());
        }
        if (change.pictureUri() != null) {
            builder.setPictureUri(change.pictureUri());
        }
        if (change.splitParts() != null) {
            builder.setSplitParts(change.splitParts());
        }
        return builder.setTimeStamp(System.currentTimeMillis()).build();
    }

    // org.totschnig.myexpenses.util.LazyFontSelector.processChar(char[],int,java.lang.StringBuffer)
    //SNIPPET_STARTS
    protected Chunk processChar(char[] cc, int k, StringBuffer sb) throws DocumentException, IOException {
        Chunk newChunk = null;
        char c = cc[k];
        if (c == '\n' || c == '\r') {
            sb.append(c);
        } else {
            Font font;
            if (Utilities.isSurrogatePair(cc, k)) {
                int u = Utilities.convertToUtf32(cc, k);
                for (int f = 0; f < files.length; ++f) {
                    font = getFont(f);
                    if (font.getBaseFont().charExists(u)
                            || Character.getType(u) == Character.FORMAT) {
                        if (currentFont != font) {
                            if (sb.length() > 0 && currentFont != null) {
                                newChunk = new Chunk(sb.toString(), currentFont);
                                sb.setLength(0);
                            }
                            currentFont = font;
                        }
                        sb.append(c);
                        sb.append(cc[++k]);
                        break;
                    }
                }
            } else {
                for (int f = 0; f < files.length; ++f) {
                    font = getFont(f);
                    if (font.getBaseFont().charExists(c)
                            || Character.getType(c) == Character.FORMAT) {
                        if (currentFont != font) {
                            if (sb.length() > 0 && currentFont != null) {
                                newChunk = new Chunk(sb.toString(), currentFont);
                                sb.setLength(0);
                            }
                            currentFont = font;
                        }
                        sb.append(c);
                        break;
                    }
                }
            }
        }
        return newChunk;
    }

    private Font getFont(int f) {
        return null;
    }

    // se.emilsjolander.stickylistheaders.StickyListHeadersListView.updateOrClearHeader(int)
    //SNIPPET_STARTS
    private void updateOrClearHeader(int firstVisiblePosition) {
        final int adapterCount = mAdapter == null ? 0 : mAdapter.getCount();
        if (adapterCount == 0 || !mAreHeadersSticky) {
            return;
        }

        final int headerViewCount = mList.getHeaderViewsCount();
        int headerPosition = firstVisiblePosition - headerViewCount;
        if (mList.getChildCount() > 0) {
            View firstItem = mList.getChildAt(0);
            if (firstItem.getBottom() < stickyHeaderTop()) {
                headerPosition++;
            }
        }

        // It is not a mistake to call getFirstVisiblePosition() here.
        // Most of the time getFixedFirstVisibleItem() should be called
        // but that does not work great together with getChildAt()
        final boolean doesListHaveChildren = mList.getChildCount() != 0;
        final boolean isFirstViewBelowTop = doesListHaveChildren
                && mList.getFirstVisiblePosition() == 0
                && mList.getChildAt(0).getTop() >= stickyHeaderTop();
        final boolean isHeaderPositionOutsideAdapterRange = headerPosition > adapterCount - 1
                || headerPosition < 0;
        if (!doesListHaveChildren || isHeaderPositionOutsideAdapterRange || isFirstViewBelowTop) {
            clearHeader();
            return;
        }

        updateHeader(headerPosition);
    }
    //SNIPPETS_END

    private void clearHeader() {

    }

    private int stickyHeaderTop() {
        return 0;
    }

    private void updateHeader(int headerPosition) {

    }

    private static class Account {
        public long grouping;


        public static Account getInstanceFromDb(long mAccountId) {
            return null;
        }
    }

    private class Intent {

        public Intent(MyExpenses myExpenses, Class<ManageCategories> manageCategoriesClass) {

        }

        public void setAction(String s) {

        }

        public void putExtra(String keyAccountid, long mAccountId) {

        }
    }

    private class Grouping {
        public static final long NONE = 1L;
    }

    private class StickyListHeadersAdapter {
    }

    private static class TransactionList {
        public static final Object KEY_FILTER = null;

        public Object getFilterCriteria() {
            return null;
        }
    }

    private class Bundle {
        public void putSparseParcelableArray(Object keyFilter, Object filterCriteria) {

        }

        public void putLong(Dummy keyRowid, long mAccountId) {

        }
    }

    private static class TaskExecutionFragment {
        public static final Object TASK_SPLIT = null;

        public static Object newInstanceWithBundle(Bundle args, Object taskPrint) {
            return null;
        }
    }

    private static class ProtectionDelegate {
        public static final Dummy ASYNC_TAG = null;
        public static final Dummy PROGRESS_TAG = null;
    }

    private static class ProgressDialogFragment {
        public static Object newInstance(Dummy progress_dialog_printing) {
            return null;
        }
    }

    private static class R {
        public static Dummy string;
    }

    private class TransactionChange {
        public boolean isCreate() {
            return false;
        }

        public Long amount() {
            return null;
        }

        public Dummy splitParts() {
            return null;
        }

        public Dummy transferAccount() {
            return null;
        }

        public Dummy parentUuid() {
            return null;
        }

        public Dummy comment() {
            return null;
        }

        public Dummy methodLabel() {
            return null;
        }

        public Dummy label() {
            return null;
        }

        public Dummy uuid() {
            return null;
        }

        public Long date() {
            return null;
        }

        public Dummy payeeName() {
            return null;
        }

        public Dummy crStatus() {
            return null;
        }

        public Dummy pictureUri() {
            return null;
        }

        public Dummy referenceNumber() {
            return null;
        }

        public Builder toBuilder() {
            return null;
        }

        public boolean isCreateOrUpdate() {
            return false;
        }

        public class Builder {

            public void setParentUuid(Dummy parentUuid) {

            }

            public void setComment(Dummy comment) {

            }

            public void setDate(Long date) {

            }

            public void setAmount(Long amount) {

            }

            public void setLabel(Dummy label) {

            }

            public void setPayeeName(Dummy payeeName) {

            }

            public void setMethodLabel(Dummy methodLabel) {

            }

            public void setTransferAccount(Dummy transferAccount) {

            }

            public void setCrStatus(Dummy crStatus) {

            }

            public void setReferenceNumber(Dummy referenceNumber) {

            }

            public void setSplitParts(Dummy splitParts) {

            }

            public void setPictureUri(Dummy pictureUri) {

            }

            public Dummy setTimeStamp(long currentTimeMillis) {
                return null;
            }
        }
    }

    private class ContentProviderOperation {
    }

    private static class Transaction {
        public long transfer_account;
        public Dummy uuid;
        public Dummy comment;
        public long methodId;
        public long payeeId;
        public long parentId;
        public Dummy referenceNumber;
        public Dummy crStatus;

        public Transaction(Dummy id, Long amount) {

        }

        public Transaction() {

        }

        public static long findByUuid(Dummy parentUuid) {
            return 0;
        }

        public void setCatId(long catId) {

        }

        public void setDate(Date date) {

        }

        public ArrayList<ContentProviderOperation> buildSaveOperations(int offset, int parentOffset, boolean b) {
            return null;
        }

        public void setPictureUri(Object parse) {

        }

        public static class CrStatus {
            public static Dummy valueOf(Dummy crStatus) {
                return null;
            }
        }
    }

    private static class Payee {
        public static long extractPayeeId(Dummy payeeName, Dummy payeeToId) {
            return 0;
        }
    }

    private class DocumentException extends Exception {
    }

    private class Chunk {
        public Chunk(String toString, Font currentFont) {

        }
    }

    private class Font {
        public Dummy getBaseFont() {
            return null;
        }
    }

    private static class Utilities {
        public static int convertToUtf32(char[] cc, int k) {
            return 0;
        }

        public static boolean isSurrogatePair(char[] cc, int k) {
            return false;
        }
    }

    private class SplitTransaction extends Transaction {
        public SplitTransaction(Object id, Long amount) {
            super();
        }
    }

    public class Dummy implements Comparable {
        public Dummy progress_dialog_printing;

        public View getChildAt(int i) {
            return null;
        }

        public Dummy beginTransaction() {
            return null;
        }

        public Dummy add(Object newInstanceWithBundle, Dummy asyncTag) {
            return null;
        }

        public void commit() {

        }

        public Dummy getId() {
            return null;
        }

        public TransactionChange build() {
            return null;
        }

        @Override
        public int compareTo(Object o) {
            return 0;
        }

        public int getFirstVisiblePosition() {
            return 0;
        }

        public int getChildCount() {
            return 4;
        }

        public int getHeaderViewsCount() {
            return 0;
        }

        public boolean charExists(char c) {
            return false;
        }

        public int getCount() {
            return 0;
        }

        public boolean charExists(int u) {
            return false;
        }
    }

    private class View {
        public int getTop() {
            return 1;
        }

        public int getBottom() {
            return 3;
        }
    }

    private class ManageCategories {
    }


    private Dummy getSupportFragmentManager() {
        return null;
    }

    private void startTaskExecution(Object taskSplit, Object[] tag, Object o, int i) {

    }

    private TransactionList getCurrentFragment() {
        return null;
    }

    private void startActivity(Intent i) {

    }

    private void recordUsage(ContribFeature feature) {

    }

    private class Transfer extends Transaction {
        public Transfer(Dummy id, Long amount) {
            super();
        }
    }

    private static class Uri {
        public static Object parse(Dummy pictureUri) {
            return null;
        }
    }
}
