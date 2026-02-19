package Siena;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.TreeMap;
import java.util.UUID;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

import java.util.Map;
import java.util.HashMap;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

class Siena2<T> {
    
    /*************   Method 27   *************/
    //SNIPPET_STARTS
    public Table addTable(Class<?> clazz) {
        if(Modifier.isAbstract(clazz.getModifiers())){
            return null;
        }
        Table table = new Table();
        ClassInfo info = ClassInfo.getClassInfo(clazz);
        table.setName(info.tableName);
        table.setType("MyISAM");
        database.addTable(table);
        
        Map<String, UniqueIndex> uniques = new HashMap<String, UniqueIndex>();
        Map<String, NonUniqueIndex> indexes = new HashMap<String, NonUniqueIndex>();
        
        for (Field field : info.allFields) {
            String[] columns = ClassInfo.getColumnNames(field);
            boolean notNull = field.getAnnotation(NotNull.class) != null;
            
            Class<?> type = field.getType();
            if(!ClassInfo.isModel(type) || (ClassInfo.isModel(type) && ClassInfo.isEmbedded(field))) {
                Column column = createColumn(clazz, field, columns[0]);
                
                if(notNull || type.isPrimitive()) {
                    column.setRequired(true);
                    
                    if(type.isPrimitive() && !ClassInfo.isId(field)) { 
                        if(type == Boolean.TYPE) {
                            column.setDefaultValue("false");
                        } else {
                            column.setDefaultValue("0");
                        }
                    }
                }
                
                Id id = field.getAnnotation(Id.class);
                if(id != null) {
                    column.setPrimaryKey(true);
                    column.setRequired(true);
                    
                    if(id.value() == Generator.AUTO_INCREMENT 
                            && (Long.TYPE == type || Long.class.isAssignableFrom(type)))
                        column.setAutoIncrement(true);
                }
                
                table.addColumn(column);
            } else {
                List<Field> keys = ClassInfo.getClassInfo(type).keys;
                
                for (int i = 0; i < columns.length; i++) {
                    Field f = keys.get(i);
                    Column column = createColumn(clazz, f, columns[i]);

                    if(notNull)
                        column.setRequired(true);
                    
                    table.addColumn(column);
                }
            }
        }

        for (Field field : info.updateFields) {
            Index index = field.getAnnotation(Index.class);
            if(index != null) {
                String[] names = index.value();
                for (String name : names) {
                    NonUniqueIndex i = indexes.get(name);
                    if(i == null) {
                        i = new NonUniqueIndex();
                        i.setName(name);
                        indexes.put(name, i);
                        table.addIndex(i);
                    }
                    fillIndex(i, field);
                }
            }
            Unique unique = field.getAnnotation(Unique.class);
            if(unique != null) {
                String[] names = unique.value();
                for (String name : names) {
                    UniqueIndex i = uniques.get(name);
                    if(i == null) {
                        i = new UniqueIndex();
                        i.setName(name);
                        uniques.put(name, i);
                        table.addIndex(i);
                    }
                    fillIndex(i, field);
                }
            }
        }
        tables.put(table.getName(), table);
        return table;
    }
    
    /*************   Method 29   *************/ 
    //SNIPPET_STARTS
    public static void encode( ByteBuffer raw, CharBuffer encoded ){
        byte[] raw3 = new byte[3];
        byte[] enc4 = new byte[4];

        while( raw.hasRemaining() ){
            int rem = Math.min(3,raw.remaining());
            raw.get(raw3,0,rem);
            Base64.encode3to4(enc4, raw3, rem, Base64.NO_OPTIONS );
            for( int i = 0; i < 4; i++ ){
                encoded.put( (char)(enc4[i] & 0xFF) );
            }
        }   
    }

    // @Override // Removed to allow compilation
    /*************   Method 30   *************/ 
    //SNIPPET_STARTS
    public int save(Iterable<?> objects) {
        List<Object> entities2Insert = new ArrayList<Object>();
        List<Object> entities2Update = new ArrayList<Object>();

        for(Object obj:objects){
            Class<?> clazz = obj.getClass();
            ClassInfo info = ClassInfo.getClassInfo(clazz);
            Field idField = info.getIdField();
            
            Object idVal = Util.readField(obj, idField);
            
            if(idVal == null){
                entities2Insert.add(obj);
            }
            else{
                entities2Update.add(obj);
            }
        }
        return insert(entities2Insert) + update(entities2Update);
    }
    
    /*************   Method 31   *************/
    //SNIPPET_STARTS
    public static Object readField(Object object, Field field) {
        boolean wasAccess = true;
        if(!field.isAccessible()){
            field.setAccessible(true);
            wasAccess = false;
        }
        try {
            return field.get(object);
        } catch (Exception e) {
            throw new SienaException(e);
        } finally {
            if(!wasAccess){
                field.setAccessible(false);
            }
        }
    }
    
    /*************   Method 40   *************/
    //SNIPPET_STARTS
    public static Entity createEntityInstance(Field idField, ClassInfo info, Object obj){
        Entity entity = null;
        Id id = idField.getAnnotation(Id.class);
        Class<?> type = idField.getType();

        if(id != null){
            switch(id.value()) {
            case NONE:
                Object idVal = null;
                idVal = Util.readField(obj, idField);
                if(idVal == null)
                    throw new SienaException("Id Field " + idField.getName() + " value null");
                String keyVal = Util.toString(idField, idVal);              
                entity = new Entity(info.tableName, keyVal);
                break;
            case AUTO_INCREMENT:
                
                if(Long.TYPE == type || Long.class.isAssignableFrom(type)){
                    entity = new Entity(info.tableName);
                }else {
                    Object idStringVal = null;
                    idStringVal = Util.readField(obj, idField);
                    if(idStringVal == null)
                        throw new SienaException("Id Field " + idField.getName() + " value null");
                    String keyStringVal = Util.toString(idField, idStringVal);              
                    entity = new Entity(info.tableName, keyStringVal);
                }
                break;
            case UUID:
                entity = new Entity(info.tableName, UUID.randomUUID().toString());
                break;
            default:
                throw new SienaRestrictedApiException("DB", "createEntityInstance", "Id Generator "+id.value()+ " not supported");
            }
        }
        else throw new SienaException("Field " + idField.getName() + " is not an @Id field");

        return entity;
    }
    
    /*************   Method 41   *************/
    //SNIPPET_STARTS
    public static void fillRequestElement(Object obj, Element element, boolean ids) {
        Class<?> clazz = obj.getClass();
        element.addAttribute("class", clazz.getName());
        
        Field[] fields = clazz.getDeclaredFields();
        for (Field field : fields) {
            if(field.getType() == Class.class) continue;
            if(ids && !ClassInfo.isId(field)) continue;
            field.setAccessible(true);
            Object value;
            try {
                value = field.get(obj);
            } catch (Exception e) {
                throw new SienaException(e);
            }
            Class<?> type = field.getType();
            if(ClassInfo.isModel(type)) {
                Element f = element.addElement("object");
                f.addAttribute("name", field.getName());
                if(value != null) {
                    fillRequestElement((Model) value, f, true);
                }
            } else {
                Element f = element.addElement("field");
                f.addAttribute("name", field.getName());
                if(value != null) {
                    f.setText(Util.toString(field, value));
                }
            }
        }
    }
    
    /*************   Method 42   *************/
    //SNIPPET_STARTS
    public static <T> int mapSelectResult(SelectResult res, Iterable<T> objects) {
        List<Item> items = res.getItems();

        Class<?> clazz = null;
        ClassInfo info = null;
        int nb = 0;
        for(T obj: objects){
            if(clazz == null){
                clazz = obj.getClass();
                info = ClassInfo.getClassInfo(clazz);               
            }
            String itemName = getItemName(clazz, obj);
            Item theItem = null;
            for(Item item:items){
                if(item.getName().equals(itemName)){
                    theItem = item;
                    items.remove(item);
                    break;
                }
            }
            if(theItem != null){
                fillModel(theItem, clazz, info, obj);
                nb++;
            }
        }
        return nb;
    }
    
    /*************   Method 43   *************/ 
    //SNIPPET_STARTS
    public void addAndMoveCursor(String cursor){
            
        if(cursorIdx < cursors.size()-1 && cursorIdx>=0){
            cursors.set(++cursorIdx, cursor);
        }
        else{
            cursors.add(++cursorIdx, cursor);
        }
    }
    
    /*************   Method 44   *************/ 
    //SNIPPET_STARTS 
    public List<T> get() {
        List<T> results;
        switch(mapType){
        case KEYS_ONLY:
            results = GaeMappingUtils.mapEntitiesKeysOnly(entities, query.getQueriedClass());
            break;
        case ALL:
        default:
            results = pm.map(query, entities);
            break;
        }
        
        QueryOptionPage pag = (QueryOptionPage)query.option(QueryOptionPage.ID);
        QueryOptionGaeContext gaeCtx = (QueryOptionGaeContext)query.option(QueryOptionGaeContext.ID);
        if(pag.isPaginating()){
            if(results.size() == 0){
                gaeCtx.noMoreDataAfter = true;
            }else {
                gaeCtx.noMoreDataAfter = false;
            }
        }
        return results;
    }
    
    /*************   Method 45   *************/ 
    //SNIPPET_STARTS
    public static void embed(ReplaceableItem item, String embeddingColumnName, Object embeddedObj){
        Class<?> clazz = embeddedObj.getClass();
        if(clazz.isArray() || Collection.class.isAssignableFrom(clazz)){
            throw new SienaException("can't serializer Array/Collection in native mode");
        }
        
        for (Field f : ClassInfo.getClassInfo(clazz).updateFields) {
            String propValue = SdbMappingUtils.objectFieldToString(embeddedObj, f);
            if(propValue != null){
                ReplaceableAttribute attr = 
                    new ReplaceableAttribute(
                            getEmbeddedAttributeName(embeddingColumnName, f), propValue, true);
                item.withAttributes(attr);
            }else {
                if (ClassInfo.isEmbeddedNative(f)){
                    SdbNativeSerializer.embed(
                        item, 
                        getEmbeddedAttributeName(embeddingColumnName, f), 
                        Util.readField(embeddedObj, f));
                }
            }
        }
    }
    
    /*************   Method 46   *************/ 
    //SNIPPET_STARTS
    public Response putAttributes(String domain, Item item) {
        TreeMap<String, String> parameters = new TreeMap<String, String>();
        parameters.put("Action", "PutAttributes");
        parameters.put("DomainName", domain);
        parameters.put("ItemName", item.name);

        int i = 0;
        for (Map.Entry<String, List<String>> entry : item.attributes.entrySet()) {
            List<String> values = entry.getValue();
            for (String value : values) {
                parameters.put("Attribute."+i+".Name", entry.getKey());
                parameters.put("Attribute."+i+".Value", value);
                parameters.put("Attribute."+i+".Replace", "true"); 
                i++;
            }
        }
        return request(parameters, new PlainHandler());
    }
    //SNIPPETS_END
    



    // STUBS ADDED BY NADEESHAN
    private class Table {
         void setName(String tableName) {}
        void setType(String string) {}
        void addColumn(Column column) {}
        void addIndex(NonUniqueIndex i) {}
        void addIndex(UniqueIndex i) {}
        String getName() {
            return "";
        }
    }

    private static class Entity {
        Entity(String tableName) { }
        Entity(String tableName, String key) { }
    }

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
    private class Database {
        void addTable(Table table) {}
    }
    private Database database = new Database();
    private class UniqueIndex{
        void setName(String name) {}
    }
    private class NonUniqueIndex{
        void setName(String name) {}
    }
    private @interface NotNull {}

    @Retention(RetentionPolicy.RUNTIME)
    private @interface ColumnAnno {
        String[] value() default {};
    }

    private class Column {
        Column(String string) {}
        // String[] value() {
        //     return new String[0];
        // }
        void setRequired(boolean b) {}
        void setDefaultValue(String string) {}
        void setPrimaryKey(boolean b) {}
        void setAutoIncrement(boolean b) {}
    }

    private enum Generator {
        AUTO_INCREMENT,
        NONE,
        UUID
    }
    private @interface Id {
        Generator value();
    }

    private @interface Index {
        String[] value();
    }

    private @interface Unique {
        String[] value();
    }

    private Column createColumn(Class<?> clazz, Field field, String string) {
        Column column = new Column(string);
        // Column annotation = field.getAnnotation(Column.class); // changed to allow compilation
        ColumnAnno annotation = field.getAnnotation(ColumnAnno.class);
        // Column annotation = new Column(c != null && c.value().length > 0 ? c.value()[0] : field.getName());
        if(annotation != null && annotation.value().length > 0){
            // can use annotation.value() to set names or other metadata
        }
        return column;
    }
    
    private void fillIndex(UniqueIndex i, Field field) {}

    private void fillIndex(NonUniqueIndex i, Field field) {}

    private Map<String, Table> tables = new HashMap<String, Table>();

    private String[] getColumnNames(Field key) {
        return new String[] {key.getName() };
    }

    private ClassInfo getClassInfo(Class<?> type) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getClassInfo'");
    }

    private boolean isModel(Class<?> type) {
        return false;
    }
    private List<Field> updateFields = new ArrayList<>();

    private static class Base64 {
        static final int NO_OPTIONS = 0;
        static void encode3to4(byte[] dest, byte[] src, int numSigBytes, int options) {}
    }

    private static class Util {
        static Object readField(Object obj, Field field) {
            return obj;
        }
        static String toString(Field field, Object value) {
            return "stubKey";
        }
    }
    

    private int update(List<Object> entities2Update) {
        return 0;
    }

    private int insert(List<Object> entities2Insert) {
        return 0;
    }

    private static class SienaException extends RuntimeException {
        SienaException(String message) { super(message); }
        SienaException(Throwable cause) { super(cause); }
    }

    private static class SienaRestrictedApiException extends RuntimeException {
        SienaRestrictedApiException(String db, String method, String msg) {
            super(db + "." + method + ": " + msg);
        }
    }
      
    private static class Element {
        void addAttribute(String name, String value) {}
        Element addElement(String name) { return new Element(); }
        void setText(String text) {}
    }

    private static class Model {}

    private static class SelectResult {
        List<Item> getItems() { return new ArrayList<>(); }
    }

    private static class Item {
        String name = "";
        Map<String, List<String>> attributes = new HashMap<>();
        String getName() { return "stubItem"; }
    }

    private static <T> String getItemName(Class<?> clazz, T obj) {
        return "stubItemName";
    }

    private static <T> void fillModel(Item item, Class<?> clazz, ClassInfo info, T obj) {}

    private List<String> cursors = new ArrayList<>();
    private int cursorIdx = -1;
    private MapType mapType = MapType.ALL;
    private List<Object> entities = new ArrayList<>();
    private Query query = new Query();
    private PM pm = new PM();

    private enum MapType {
        KEYS_ONLY,
        ALL
    }

    private static class Query {
        Class<?> getQueriedClass() { return Object.class; }
        Object option(Object id) { return id; } // returns a QueryOption stub
    }

    private static class QueryOptionPage {
        static final Object ID = new Object();
        boolean isPaginating() { return false; }
    }

    private static class QueryOptionGaeContext {
        static final Object ID = new Object();
        boolean noMoreDataAfter;
    }

    private static class PM {
        <T> List<T> map(Query query, List<Object> entities) {
            return new ArrayList<>();
        }
    }

    private static class GaeMappingUtils {
        static <T> List<T> mapEntitiesKeysOnly(List<Object> entities, Class<?> cls) {
            return new ArrayList<>();
        }
    }

    private static class ReplaceableItem {
        ReplaceableItem withAttributes(ReplaceableAttribute attr) {
            return this;
        }
    }

    private static class ReplaceableAttribute {
        ReplaceableAttribute(String name, String value, boolean flag) {}
    }

    private static class SdbMappingUtils {
        static String objectFieldToString(Object obj, Field f) {
            try {
                Object val = f.get(obj);
                return val != null ? val.toString() : "";
            } catch (Exception e) {
                return "";
            }
        }
    }

    private static class SdbNativeSerializer {
        static void embed(ReplaceableItem item, String columnName, Object embeddedObj) {}
    }

    private static String getEmbeddedAttributeName(String parent, Field f) {
        return parent + "_" + f.getName();
    }

    private static class Response{}
    private static class PlainHandler {}
    private Response request(Map<String, String> params, PlainHandler handler) {
        return new Response();
    }

}