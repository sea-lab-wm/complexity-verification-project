getConnection csds con1_2 Connection 
con1_2 con1 assertSame 
getConnection csds con2 Connection 
con2 con1 assertNotSame 
con1 stopCloseSuppression csds 
con1 shouldClose csds assertTrue 
con1_1 
con1_2 
close con1 
con2 shouldClose csds assertTrue 
con thenReturn getConnection ds when 
close con2 
close con 
con thenReturn getConnection ds when 
close con 
ds ExtendedConnectionDataSourceProxy csds ExtendedConnectionDataSourceProxy 
getConnection csds con1 Connection 
con1 startCloseSuppression csds 
getConnection csds con1_1 Connection 
con1_1 con1 assertSame 
ds_6_snip_3$SpringBatch_testOperationWithDirectCloseCall 
testOperationWithDirectCloseCall SQLException 
Connection mock con Connection 
DataSource mock ds DataSource 
close con1_1 
