    void Constraint(HsqlName name, int[] mainCols, Table refTable, int[] refCols, // Added return type void to allow compilation
               int type, int deleteAction, int updateAction) {

        core              = new ConstraintCore();
        constName         = name;
        constType         = type;
    } // Added to allow compilation
