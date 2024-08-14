jdbcUrl checkNotNull Preconditions 
keyGeneratorType debug logger 
columnNames checkNotNull Preconditions 
context doConfigure 
columnNames split DEFAULT_COLUMNS_DELIMITER on Splitter s String 
s add colNames 
headersStr isNullOrEmpty Strings 
headersStr split DEFAULT_COLUMNS_DELIMITER on Splitter s String 
s add headers 
CONFIG_JDBC_URL FlumeConstants getString context ipJdbcURL String 
keyGeneratorType isNullOrEmpty Strings 
batchSize CONFIG_BATCHSIZE FlumeConstants DEFAULT_BATCH_SIZE FlumeConstants getInteger context 
CONFIG_COLUMN_NAMES getString context columnNames String 
keyGenerator toUpperCase keyGeneratorType valueOf DefaultKeyGenerator 
CONFIG_HEADER_NAMES getString context headersStr String 
autoGenerateKey 
CONFIG_ROWKEY_TYPE_GENERATOR getString context keyGeneratorType String 
iae IllegalArgumentException 
fullTableName checkNotNull Preconditions 
keyGeneratorType values DefaultKeyGenerator error logger 
zookeeperQuorum isNullOrEmpty Strings 
iae propagate Throwables 
jdbcUrl zookeeperQuorum getUrl QueryUtil 
ipJdbcURL isNullOrEmpty Strings 
ds_6_snip_3$Phoenix_configure 
configure context Context 
createTableDdl CONFIG_TABLE_DDL FlumeConstants getString context 
fullTableName CONFIG_TABLE FlumeConstants getString context 
CONFIG_ZK_QUORUM FlumeConstants getString context zookeeperQuorum String 
jdbcUrl debug logger 
jdbcUrl ipJdbcURL 
toString colNames debug logger 
headersStr debug logger 
