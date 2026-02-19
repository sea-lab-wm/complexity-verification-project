package Siena;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.lang.reflect.Field;

public class Siena1 {

    /*************   Method 28   *************/ 
    //SNIPPET_STARTS
    public List<String> getUpdateFieldsColumnNames() {
        List<String> strs = new ArrayList<String>(this.updateFields.size());
        for(Field field: this.updateFields){
            Column c = field.getAnnotation(Column.class);
            if(c != null && c.value().length > 0) {
                strs.add(c.value()[0]);
            }

            else if(isModel(field.getType())) {
                ClassInfo ci = getClassInfo(field.getType());
                for (Field key : ci.keys) {
                    Collections.addAll(strs, getColumnNames(key));
                }
            }
            else {
                strs.add(field.getName());
            }
        }
        return strs;
    }
    //SNIPPETS_END
    // STUBS ADDED BY NADEESHAN
    private List<Field> updateFields = new ArrayList<>();

    private static class ClassInfo {
        static ClassInfo getClassInfo(Class<?> clazz) {
            return new ClassInfo(); 
        }
        String tableName = "";
        Field[] allFields = new Field[0];
        Field[] updateFields = new Field[0];
        List<Field> keys = new ArrayList<>();
        static String[] getColumnNames(Field field) {
            return new String[] { field.getName() }; 
        }
        static boolean isModel(Class<?> type) {
            return false;
        }
        static boolean isId(Field field) {
            return false;
        }
        static boolean isEmbedded(Field field) {
            return false;
        }
        Field getIdField() {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'getIdField'");
        }
        static boolean isEmbeddedNative(Field f) {
            return false;
        }

    }

    private boolean isModel(Class<?> type) {
        return false;
    }

    private ClassInfo getClassInfo(Class<?> type) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getClassInfo'");
    }

    private String[] getColumnNames(Field key) {
        return new String[] {key.getName() };
    }

    public @interface Column {
        String[] value();
    }
}
