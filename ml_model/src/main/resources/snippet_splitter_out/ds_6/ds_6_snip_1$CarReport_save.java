package snippet_splitter_out.ds_6;

public class ds_6_snip_1$CarReport_save {
  // me.kuehle.carreport.gui.dialog.EditFuelTypeDialogFragment.save()
  // SNIPPET_STARTS
  private boolean save() {
    FormValidator validator = new FormValidator();
    validator.add(new FormFieldNotEmptyValidator(mEdtName));
    validator.add(
        new AbstractFormFieldValidator(mEdtName) {

          // @Override // Removed to allow compilation
          protected boolean isValid() {
            String name = mEdtName.getText().toString();
            return !mOtherFuelTypeNames.contains(name);
          }

          // @Override // Removed to allow compilation
          protected int getMessage() {
            return R.string.validate_error_fuel_type_exists;
          }
        });
    validator.add(new FormFieldNotEmptyValidator(mEdtCategory));
    if (validator.validate()) {
      FuelTypeContentValues values = new FuelTypeContentValues();
      values.putName(mEdtName.getText().toString());
      values.putCategory(mEdtCategory.getText().toString());
      if (mFuelType == null) {
        values.insert(getActivity().getContentResolver());
      } else {
        FuelTypeSelection where = new FuelTypeSelection().id(mFuelType.getId());
        values.update(getActivity().getContentResolver(), where);
      }
      return true;
    } else {
      return false;
    }
  }
}
