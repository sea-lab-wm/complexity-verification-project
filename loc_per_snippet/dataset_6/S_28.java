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

    // org.hibernate.pretty.MessageHelper.collectionInfoString(org.hibernate.persister.collection.CollectionPersister,org.hibernate.collection.spi.PersistentCollection,java.io.Serializable,org.hibernate.engine.spi.SharedSessionContractImplementor)
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
