package snippet_splitter_out.ds_6;

public class ds_6_snip_1$MyExpenses_contribFeatureCalled {
  // org.totschnig.myexpenses.activity.MyExpenses.contribFeatureCalled(org.totschnig.myexpenses.model.ContribFeature,java.io.Serializable)
  // @Override // removed to allow compilation
  // SNIPPET_STARTS
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
          startTaskExecution(TaskExecutionFragment.TASK_SPLIT, (Object[]) tag, null, 0);
        }
        break;
      case PRINT:
        TransactionList tl = getCurrentFragment();
        if (tl != null) {
          Bundle args = new Bundle();
          args.putSparseParcelableArray(TransactionList.KEY_FILTER, tl.getFilterCriteria());
          args.putLong(KEY_ROWID, mAccountId);
          getSupportFragmentManager()
              .beginTransaction()
              .add(
                  TaskExecutionFragment.newInstanceWithBundle(args, TASK_PRINT),
                  ProtectionDelegate.ASYNC_TAG)
              .add(
                  ProgressDialogFragment.newInstance(R.string.progress_dialog_printing),
                  ProtectionDelegate.PROGRESS_TAG)
              .commit();
        }
        break;
    }
  }
}
