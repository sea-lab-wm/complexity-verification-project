sessionImpl Session session Session 
e 
beginTransaction session transaction Transaction 
dialect DerbyDialect 
queryString prepareStatement getStatementPreparer getJdbcCoordinator sessionImpl query PreparedStatement 
value 
query extract getResultSetReturn getJdbcCoordinator sessionImpl resultSet ResultSet 
next resultSet 
value getLong resultSet 
close resultSet 
WorkImpl work WorkImpl 
commit transaction 
ds_6_snip_4$HibernateORM_extractSequenceValue 
extractSequenceValue sessionImpl SessionImplementor 
Work WorkImpl 
value 
work doWork sessionImpl Session 
e GenericJDBCException 
value work 
rollback transaction 
execute connection Connection SQLException 
