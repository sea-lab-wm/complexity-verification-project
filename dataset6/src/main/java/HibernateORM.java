import javax.persistence.Column;
import java.io.Serializable;
import java.sql.*;

public class HibernateORM {
    private final String DEFAULT_BATCH_SIZE = "null";
    private Object targetedPersister;
    private Update[] updates;
    private String idInsertSelect;
    private ParameterSpecification[] idSelectParameterSpecifications;
    private ParameterSpecification[][] assignmentParameterSpecifications;
    private Update queryString;
    private Object dialect;

    //ADDED BY KOBI
    public void runAll() {
        getOverriddenColumn("propertyName");
        new TimesTenDialect();
        try {
            execute(new SharedSessionContractImplementor(), new QueryParameters());
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        extractSequenceValue(new SessionImplementor());
        collectionInfoString(new CollectionPersister(), new PersistentCollection(), new Serializable() {}, new SharedSessionContractImplementor());
    }

    // s26: org.hibernate.cfg.AbstractPropertyHolder.getOverriddenColumn(java.lang.String)
    /**
     * Get column overriding, property first, then parent, then holder
     * replace the placeholder 'collection&&element' with nothing
     *
     * These rules are here to support both JPA 2 and legacy overriding rules.
     */
//    @Override // Removed to allow compilation
    //SNIPPET_STARTS
    public Column[] getOverriddenColumn(String propertyName) {
        Column[] result = getExactOverriddenColumn( propertyName );
        if (result == null) {
            //the commented code can be useful if people use the new prefixes on old mappings and vice versa
            // if we enable them:
            // WARNING: this can conflict with user's expectations if:
            //  - the property uses some restricted values
            //  - the user has overridden the column
            // also change getOverriddenJoinColumn and getOverriddenJoinTable as well

//			if ( propertyName.contains( ".key." ) ) {
//				//support for legacy @AttributeOverride declarations
//				//TODO cache the underlying regexp
//				result = getExactOverriddenColumn( propertyName.replace( ".key.", ".index."  ) );
//			}
//			if ( result == null && propertyName.endsWith( ".key" ) ) {
//				//support for legacy @AttributeOverride declarations
//				//TODO cache the underlying regexp
//				result = getExactOverriddenColumn(
//						propertyName.substring( 0, propertyName.length() - ".key".length() ) + ".index"
//						);
//			}
//			if ( result == null && propertyName.contains( ".value." ) ) {
//				//support for legacy @AttributeOverride declarations
//				//TODO cache the underlying regexp
//				result = getExactOverriddenColumn( propertyName.replace( ".value.", ".element."  ) );
//			}
//			if ( result == null && propertyName.endsWith( ".value" ) ) {
//				//support for legacy @AttributeOverride declarations
//				//TODO cache the underlying regexp
//				result = getExactOverriddenColumn(
//						propertyName.substring( 0, propertyName.length() - ".value".length() ) + ".element"
//						);
//			}
            if ( result == null && propertyName.contains( ".collection&&element." ) ) {
                //support for non map collections where no prefix is needed
                //TODO cache the underlying regexp
                result = getExactOverriddenColumn( propertyName.replace( ".collection&&element.", "."  ) );
            }
        }
        return result;
    }

    private static class CollectionPersister {
        public char[] getRole() {
            return new char[0];
        }

        public CollectionPersister getOwnerEntityPersister() {
            throw new Error();
        }

        public Type getIdentifierType() {
            throw new Error();
        }
    }

    private static class PersistentCollection {
        public Object getOwner() {
            throw new Error();
        }
    }

    private static class Type {
        public Class<?> getReturnedClass() {
            throw new Error();
        }

        public char[] toLoggableString(Serializable ownerKey, Object factory) {
            return new char[0];
        }
    }

    private static class EntityEntry {
        public Serializable getId() {
            throw new Error();
        }
    }

    public class TimesTenDialect extends Dialect {
        // s27: org.hibernate.dialect.TimesTenDialect.TimesTenDialect()
        /**
         * Constructs a TimesTenDialect
         */
        //SNIPPET_STARTS
        public TimesTenDialect() {
            super();
            registerColumnType(Types.BIT, "TINYINT");
            registerColumnType(Types.BIGINT, "BIGINT");
            registerColumnType(Types.SMALLINT, "SMALLINT");
            registerColumnType(Types.TINYINT, "TINYINT");
            registerColumnType(Types.INTEGER, "INTEGER");
            registerColumnType(Types.CHAR, "CHAR(1)");
            registerColumnType(Types.VARCHAR, "VARCHAR($l)");
            registerColumnType(Types.FLOAT, "FLOAT");
            registerColumnType(Types.DOUBLE, "DOUBLE");
            registerColumnType(Types.DATE, "DATE");
            registerColumnType(Types.TIME, "TIME");
            registerColumnType(Types.TIMESTAMP, "TIMESTAMP");
            registerColumnType(Types.VARBINARY, "VARBINARY($l)");
            registerColumnType(Types.NUMERIC, "DECIMAL($p, $s)");
            // TimesTen has no BLOB/CLOB support, but these types may be suitable
            // for some applications. The length is limited to 4 million bytes.
            registerColumnType(Types.BLOB, "VARBINARY(4000000)");
            registerColumnType(Types.CLOB, "VARCHAR(4000000)");

            getDefaultProperties().setProperty(Environment.USE_STREAMS_FOR_BINARY, "true");
            getDefaultProperties().setProperty(Environment.STATEMENT_BATCH_SIZE, DEFAULT_BATCH_SIZE);
            registerFunction("lower", new StandardSQLFunction("lower"));
            registerFunction("upper", new StandardSQLFunction("upper"));
            registerFunction("rtrim", new StandardSQLFunction("rtrim"));
            registerFunction("concat", new StandardSQLFunction("concat", StandardBasicTypes.STRING));
            registerFunction("mod", new StandardSQLFunction("mod"));
            registerFunction("to_char", new StandardSQLFunction("to_char", StandardBasicTypes.STRING));
            registerFunction("to_date", new StandardSQLFunction("to_date", StandardBasicTypes.TIMESTAMP));
            registerFunction("sysdate", new NoArgSQLFunction("sysdate", StandardBasicTypes.TIMESTAMP, false));
            registerFunction("getdate", new NoArgSQLFunction("getdate", StandardBasicTypes.TIMESTAMP, false));
            registerFunction("nvl", new StandardSQLFunction("nvl"));
        }
    }

    // s28: org.hibernate.hql.spi.id.TableBasedUpdateHandlerImpl.execute(org.hibernate.engine.spi.SharedSessionContractImplementor,org.hibernate.engine.spi.QueryParameters)
//    @Override // Removed to allow compilation
    //SNIPPET_STARTS
    public int execute(SharedSessionContractImplementor session, QueryParameters queryParameters) throws Exception { // throws Exception added to allow compilation
        prepareForUse( targetedPersister, session );
        try {
            // First, save off the pertinent ids, as the return value
            PreparedStatement ps = null;
            int resultCount = 0;
            try {
                try {
                    ps = session.getJdbcCoordinator().getStatementPreparer().prepareStatement( idInsertSelect, false );
                    int position = 1;
                    position += handlePrependedParametersOnIdSelection( ps, session, position );
                    for ( ParameterSpecification parameterSpecification : idSelectParameterSpecifications ) {
                        position += parameterSpecification.bind( ps, queryParameters, session, position );
                    }
                    resultCount = session.getJdbcCoordinator().getResultSetReturn().executeUpdate( ps );
                }
                finally {
                    if ( ps != null ) {
                        session.getJdbcCoordinator().getLogicalConnection().getResourceRegistry().release( ps );
                        session.getJdbcCoordinator().afterStatementExecution();
                    }
                }
            }
            catch( SQLException e ) {
                throw session.getJdbcServices().getSqlExceptionHelper().convert( e, "could not insert/select ids for bulk update", idInsertSelect );
            }

            // Start performing the updates
            for ( int i = 0; i < updates.length; i++ ) {
                if ( updates[i] == null ) {
                    continue;
                }
                try {
                    try {
                        ps = session.getJdbcCoordinator().getStatementPreparer().prepareStatement( updates[i], false );
                        if ( assignmentParameterSpecifications[i] != null ) {
                            int position = 1; // jdbc params are 1-based
                            for ( int x = 0; x < assignmentParameterSpecifications[i].length; x++ ) {
                                position += assignmentParameterSpecifications[i][x].bind( ps, queryParameters, session, position );
                            }
                            handleAddedParametersOnUpdate( ps, session, position );
                        }
                        session.getJdbcCoordinator().getResultSetReturn().executeUpdate( ps );
                    }
                    finally {
                        if ( ps != null ) {
                            session.getJdbcCoordinator().getLogicalConnection().getResourceRegistry().release( ps );
                            session.getJdbcCoordinator().afterStatementExecution();
                        }
                    }
                }
                catch( SQLException e ) {
                    throw session.getJdbcServices().getSqlExceptionHelper().convert( e, "error performing bulk update", updates[i] );
                }
            }

            return resultCount;
        }
        finally {
            releaseFromUse( targetedPersister, session );
        }
    }

    private void handleAddedParametersOnUpdate(PreparedStatement ps, SharedSessionContractImplementor session, int position) {

    }

    private int handlePrependedParametersOnIdSelection(PreparedStatement ps, SharedSessionContractImplementor session, int position) {
        return 0;
    }

    private void prepareForUse(Object targetedPersister, SharedSessionContractImplementor session) {

    }

    private void releaseFromUse(Object targetedPersister, SharedSessionContractImplementor session) {

    }

    // s29: org.hibernate.id.SequenceValueExtractor.extractSequenceValue(org.hibernate.engine.spi.SessionImplementor)
    //SNIPPET_STARTS
    public long extractSequenceValue(final SessionImplementor sessionImpl) {
        class WorkImpl implements Work {
            private long value;

            public void execute(Connection connection) throws SQLException {
                Session session = (Session) sessionImpl;
                Transaction transaction = session.beginTransaction();
                try {
                    final PreparedStatement query = sessionImpl.getJdbcCoordinator()
                            .getStatementPreparer()
                            .prepareStatement( queryString );
                    ResultSet resultSet = sessionImpl.getJdbcCoordinator().getResultSetReturn().extract( query );
                    resultSet.next();
                    value = resultSet.getLong( 1 );

                    resultSet.close();
                    transaction.commit();
                }catch (GenericJDBCException e){
                    transaction.rollback();
                    throw e;
                }
                if ( dialect instanceof DerbyDialect ) {
                    value--;
                }
            }
        }
        WorkImpl work = new WorkImpl();
        ((Session) sessionImpl).doWork( work );
        return work.value;
    }

    // s30: org.hibernate.pretty.MessageHelper.collectionInfoString(org.hibernate.persister.collection.CollectionPersister,org.hibernate.collection.spi.PersistentCollection,java.io.Serializable,org.hibernate.engine.spi.SharedSessionContractImplementor)
    /**
     * Generate an info message string relating to a particular managed
     * collection.  Attempts to intelligently handle property-refs issues
     * where the collection key is not the same as the owner key.
     *
     * @param persister The persister for the collection
     * @param collection The collection itself
     * @param collectionKey The collection key
     * @param session The session
     * @return An info string, in the form [Foo.bars#1]
     */
    //SNIPPET_STARTS
    public static String collectionInfoString(
            CollectionPersister persister,
            PersistentCollection collection,
            Serializable collectionKey,
            SharedSessionContractImplementor session ) {

        StringBuilder s = new StringBuilder();
        s.append( '[' );
        if ( persister == null ) {
            s.append( "<unreferenced>" );
        }
        else {
            s.append( persister.getRole() );
            s.append( '#' );

            Type ownerIdentifierType = persister.getOwnerEntityPersister()
                    .getIdentifierType();
            Serializable ownerKey;
            // TODO: Is it redundant to attempt to use the collectionKey,
            // or is always using the owner id sufficient?
            if ( collectionKey.getClass().isAssignableFrom(
                    ownerIdentifierType.getReturnedClass() ) ) {
                ownerKey = collectionKey;
            }
            else {
                Object collectionOwner = collection == null ? null : collection.getOwner();
                EntityEntry entry = collectionOwner == null ? null : session.getPersistenceContext().getEntry(collectionOwner);
                ownerKey = entry == null ? null : entry.getId();
            }
            s.append( ownerIdentifierType.toLoggableString(
                    ownerKey, session.getFactory() ) );
        }
        s.append( ']' );

        return s.toString();
    }
    //SNIPPETS_END

    private class Dialect {
    }

    private Column[] getExactOverriddenColumn(String replace) {
        return new Column[0];
    }

    private class StandardSQLFunction {
        public StandardSQLFunction(String lower) {
        }

        public StandardSQLFunction(String concat, Object string) {

        }

        public StandardSQLFunction() {

        }
    }

    private class NoArgSQLFunction extends StandardSQLFunction {
        public NoArgSQLFunction(String sysdate, Object p1, boolean b) {
            super();
        }
    }

    private static class StandardBasicTypes {
        public static final Object STRING = null;
        public static final Object TIMESTAMP = null;
    }

    public static class Environment {
        public static final Object USE_STREAMS_FOR_BINARY = null;
        public static final Object STATEMENT_BATCH_SIZE = null;

        public void setProperty(Object useStreamsForBinary, String aTrue) {

        }

        public Environment getStatementPreparer() {
            throw new Error();
        }

        public PreparedStatement prepareStatement(String idInsertSelect, boolean b) {
            throw new Error();
        }

        public PreparedStatement prepareStatement(Update idInsertSelect, boolean b) {
            throw new Error();
        }

        public void afterStatementExecution() throws SQLException{

        }

        public Environment getLogicalConnection() {
            throw new Error();
        }

        public Environment getResultSetReturn() {
            throw new Error();
        }

        public int executeUpdate(PreparedStatement ps) {
            return 0;
        }

        public Environment getResourceRegistry() {
            throw new Error();
        }

        public void release(PreparedStatement ps) {

        }

        public Environment getSqlExceptionHelper() {
            throw new Error();
        }

        public Exception convert(SQLException e, String s, String idInsertSelect) {
            throw new Error();
        }
        public Exception convert(SQLException e, String s, Update idInsertSelect) {
            throw new Error();
        }

        public PreparedStatement prepareStatement(Update queryString) {
            throw new Error();
        }

        public ResultSet extract(PreparedStatement query) {
            throw new Error();
        }

        public EntityEntry getEntry(Object collectionOwner) {
            throw new Error();
        }

        public Object convert(long l, Object nanoseconds) {
            throw new Error();
        }
    }


    private void registerFunction(String lower, StandardSQLFunction lower1) {

    }

    private Environment getDefaultProperties() {
        throw new Error();
    }

    private void registerColumnType(int bit, String tinyint) {

    }

    private class SharedSessionContractImplementor {
        public Environment getJdbcCoordinator() {
            throw new Error();
        }

        public Environment getJdbcServices() {
            throw new Error();
        }

        public Environment getPersistenceContext() {
            throw new Error();
        }

        public Object getFactory() {
            throw new Error();
        }
    }

    private class QueryParameters {
    }

    private class SessionImplementor extends Session {
        public Environment getJdbcCoordinator() {
            throw new Error();
        }
    }

    private interface Work {

    }

    private class Transaction {
        public void commit() {

        }

        public void rollback() {

        }
    }

    private class GenericJDBCException extends SQLException {
    }

    private class DerbyDialect {
    }

    private class Update {
    }

    private class ParameterSpecification {
        public int bind(PreparedStatement ps, QueryParameters queryParameters, SharedSessionContractImplementor session, int position) {
            return 0;
        }
    }

    private class Session {
        public Transaction beginTransaction() {
            throw new Error();
        }
        public <WorkImpl> void doWork(WorkImpl work) {

        }
    }
}
