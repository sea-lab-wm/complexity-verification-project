package FeatureExtraction.snippet_splitter_out;
public class ds_6_snip_3$SpringBatch_testOperationWithDirectCloseCall {
// org.springframework.batch.item.database.ExtendedConnectionDataSourceProxyTests.testOperationWithDirectCloseCall()
// @Test // removed to allow compilation
// SNIPPET_STARTS
public void testOperationWithDirectCloseCall() throws SQLException {
    Connection con = mock(Connection.class);
    DataSource ds = mock(DataSource.class);
    // con1
    when(ds.getConnection()).thenReturn(con);
    con.close();
    // con2
    when(ds.getConnection()).thenReturn(con);
    con.close();
    final ExtendedConnectionDataSourceProxy csds = new ExtendedConnectionDataSourceProxy(ds);
    Connection con1 = csds.getConnection();
    csds.startCloseSuppression(con1);
    Connection con1_1 = csds.getConnection();
    assertSame("should be same connection", con1_1, con1);
    // no mock call for this - should be suppressed
    con1_1.close();
    Connection con1_2 = csds.getConnection();
    assertSame("should be same connection", con1_2, con1);
    Connection con2 = csds.getConnection();
    assertNotSame("shouldn't be same connection", con2, con1);
    csds.stopCloseSuppression(con1);
    assertTrue("should be able to close connection", csds.shouldClose(con1));
    con1_1 = null;
    con1_2 = null;
    con1.close();
    assertTrue("should be able to close connection", csds.shouldClose(con2));
    con2.close();
}
}