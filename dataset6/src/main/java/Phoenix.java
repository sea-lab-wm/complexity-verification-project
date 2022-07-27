import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.client.Mutation;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hbase.thirdparty.com.google.common.base.Splitter;
import org.apache.hbase.thirdparty.com.google.common.base.Strings;
import org.apache.hbase.thirdparty.com.google.common.base.Throwables;
import org.apache.hbase.thirdparty.io.netty.channel.ChannelException;
import org.apache.hbase.thirdparty.org.apache.commons.cli.*;

import java.sql.SQLException;
import java.util.List;

//ADDED BY KOBI
import java.util.ArrayList;

public class Phoenix {

    private static final org.apache.commons.cli.Option INPUT_PATH_OPT = null;
    private static final Object MIN_TABLE_TIMESTAMP = null;
    private static final Object CONFIG_COLUMN_NAMES = null;
    private static final Object CONFIG_HEADER_NAMES = null;
    private static final Object CONFIG_ROWKEY_TYPE_GENERATOR = null;
    private static final String DEFAULT_COLUMNS_DELIMITER = "";
    private static final org.apache.commons.cli.Option TABLE_NAME_OPT = null;
    private static final org.apache.commons.cli.Option HELP_OPT = null;
    private Dummy env;
    private Dummy PLong;
    private String createTableDdl;
    private String fullTableName;
    private int batchSize;
    private Object jdbcUrl;
    private Antlr4Master.ATNConfigSet colNames;
    private Antlr4Master.ATNConfigSet headers;
    private Object keyGenerator;
    private boolean autoGenerateKey;
    private Pom.POSIXHandler logger;
    private Dummy sinkCounter;
    private Dummy serializer;

    //ADDED BY KOBI
    public void runAll() {
        try {
            doDropSchema(1, "schemaName", new byte[5], new ArrayList<Mutation>(), new ArrayList<ImmutableBytesPtr>());
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        evaluate(new Tuple(), new ImmutableBytesWritable());
        configure(new Context());
        try {
            process();
        } catch (Phoenix.EventDeliveryException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        parseOptions(new String[5]);
    }

    // org.apache.phoenix.coprocessor.MetaDataEndpointImpl.doDropSchema(long,java.lang.String,byte[],java.util.List<org.apache.hadoop.hbase.client.Mutation>,java.util.List<org.apache.phoenix.hbase.index.util.ImmutableBytesPtr>)
    //SNIPPET_STARTS
    private MetaDataMutationResult doDropSchema(long clientTimeStamp, String schemaName, byte[] key,
                                                List<Mutation> schemaMutations, List<ImmutableBytesPtr> invalidateList) throws Exception {
        PSchema schema = loadSchema(env, key, new ImmutableBytesPtr(key), clientTimeStamp, clientTimeStamp);
        boolean areTablesExists = false;
        if (schema == null) { return new MetaDataMutationResult(MutationCode.SCHEMA_NOT_FOUND,
                EnvironmentEdgeManager.currentTimeMillis(), null); }
        if (schema.getTimeStamp() < clientTimeStamp) {
            Region region = env.getRegion();
            Scan scan = MetaDataUtil.newTableRowsScan(SchemaUtil.getKeyForSchema(null, schemaName), MIN_TABLE_TIMESTAMP,
                    clientTimeStamp);
            List<Cell> results = Lists.newArrayList();
            try (RegionScanner scanner = region.getScanner(scan);) {
                scanner.next(results);
                if (results.isEmpty()) { // Should not be possible
                    return new MetaDataMutationResult(MutationCode.SCHEMA_NOT_FOUND,
                            EnvironmentEdgeManager.currentTimeMillis(), null);
                }
                do {
                    Cell kv = results.get(0);
                    if (Bytes.compareTo(kv.getRowArray(), kv.getRowOffset(), kv.getRowLength(), key, 0,
                            key.length) != 0) {
                        areTablesExists = true;
                        break;
                    }
                    results.clear();
                    scanner.next(results);
                } while (!results.isEmpty());
            }
            if (areTablesExists) { return new MetaDataMutationResult(MutationCode.TABLES_EXIST_ON_SCHEMA, schema,
                    EnvironmentEdgeManager.currentTimeMillis()); }

            return new MetaDataMutationResult(MutationCode.SCHEMA_ALREADY_EXISTS, schema,
                    EnvironmentEdgeManager.currentTimeMillis());
        }
        return new MetaDataMutationResult(MutationCode.SCHEMA_NOT_FOUND, EnvironmentEdgeManager.currentTimeMillis(),
                null);

    }

    private PSchema loadSchema(Dummy env, byte[] key, ImmutableBytesPtr immutableBytesPtr, long clientTimeStamp, long clientTimeStamp1) {
        return null;
    }

    // org.apache.phoenix.expression.ModulusExpression.evaluate(org.apache.phoenix.schema.tuple.Tuple,org.apache.hadoop.hbase.io.ImmutableBytesWritable)

//    @Override // removed to allow compilation
    //SNIPPET_STARTS
    public boolean evaluate(Tuple tuple, ImmutableBytesWritable ptr) {
        // get the dividend
        Expression dividendExpression = getDividendExpression();
        if (!dividendExpression.evaluate(tuple, ptr)) {
            return false;
        }
        if (ptr.getLength() == 0) {
            return true;
        }
        long dividend = dividendExpression.getDataType().getCodec().decodeLong(ptr, dividendExpression.getSortOrder());

        // get the divisor
        Expression divisorExpression = getDivisorExpression();
        if (!divisorExpression.evaluate(tuple, ptr)) {
            return false;
        }
        if (ptr.getLength() == 0) {
            return true;
        }
        long divisor = divisorExpression.getDataType().getCodec().decodeLong(ptr, divisorExpression.getSortOrder());

        // actually perform modulus
        long remainder = dividend % divisor;

        // return the result, use encodeLong to avoid extra Long allocation
        byte[] resultPtr=new byte[PLong.INSTANCE.getByteSize()];
        getDataType().getCodec().encodeLong(remainder, resultPtr, 0);
        ptr.set(resultPtr);
        return true;
    }

    private Expression getDataType() {
        return null;
    }

    private Expression getDivisorExpression() {
        return null;
    }

    private Expression getDividendExpression() {
        return null;
    }

    // org.apache.phoenix.flume.serializer.BaseEventSerializer.configure(org.apache.flume.Context)
//    @Override // removed to allow compilation
    //SNIPPET_STARTS
    public void configure(Context context) {

        this.createTableDdl = context.getString(FlumeConstants.CONFIG_TABLE_DDL);
        this.fullTableName = context.getString(FlumeConstants.CONFIG_TABLE);
        final String zookeeperQuorum = context.getString(FlumeConstants.CONFIG_ZK_QUORUM);
        final String ipJdbcURL = context.getString(FlumeConstants.CONFIG_JDBC_URL);
        this.batchSize = context.getInteger(FlumeConstants.CONFIG_BATCHSIZE, FlumeConstants.DEFAULT_BATCH_SIZE);
        final String columnNames = context.getString(CONFIG_COLUMN_NAMES);
        final String headersStr = context.getString(CONFIG_HEADER_NAMES);
        final String keyGeneratorType = context.getString(CONFIG_ROWKEY_TYPE_GENERATOR);

        Preconditions.checkNotNull(this.fullTableName,"Table name cannot be empty, please specify in the configuration file");
        if(!Strings.isNullOrEmpty(zookeeperQuorum)) {
            this.jdbcUrl = QueryUtil.getUrl(zookeeperQuorum);
        }
        if(!Strings.isNullOrEmpty(ipJdbcURL)) {
            this.jdbcUrl = ipJdbcURL;
        }
        Preconditions.checkNotNull(this.jdbcUrl,"Please specify either the zookeeper quorum or the jdbc url in the configuration file");
        Preconditions.checkNotNull(columnNames,"Column names cannot be empty, please specify in configuration file");
        for(String s : Splitter.on(DEFAULT_COLUMNS_DELIMITER).split(columnNames)) {
            colNames.add(s);
        }

        if(!Strings.isNullOrEmpty(headersStr)) {
            for(String s : Splitter.on(DEFAULT_COLUMNS_DELIMITER).split(headersStr)) {
                headers.add(s);
            }
        }

        if(!Strings.isNullOrEmpty(keyGeneratorType)) {
            try {
                keyGenerator =  DefaultKeyGenerator.valueOf(keyGeneratorType.toUpperCase());
                this.autoGenerateKey = true;
            } catch(IllegalArgumentException iae) {
                logger.error("An invalid key generator {} was specified in configuration file. Specify one of {}",keyGeneratorType,DefaultKeyGenerator.values());
                Throwables.propagate(iae);
            }
        }

        logger.debug(" the jdbcUrl configured is {}",jdbcUrl);
        logger.debug(" columns configured are {}",colNames.toString());
        logger.debug(" headers configured are {}",headersStr);
        logger.debug(" the keyGenerator configured is {} ",keyGeneratorType);

        doConfigure(context);

    }

    // org.apache.phoenix.flume.sink.PhoenixSink.process()
//    @Override // removed to allow compilation
    //SNIPPET_STARTS
    public Status process() throws EventDeliveryException {

        Status status = Status.READY;
        Channel channel = getChannel();
        Transaction transaction = null;
        List<Event>  events = Lists.newArrayListWithExpectedSize(this.batchSize);
        long startTime = System.nanoTime();
        try {
            transaction = channel.getTransaction();
            transaction.begin();

            for(long i = 0; i < this.batchSize; i++) {
                Event event = channel.take();
                if(event == null){
                    status = Status.BACKOFF;
                    if (i == 0) {
                        sinkCounter.incrementBatchEmptyCount();
                    } else {
                        sinkCounter.incrementBatchUnderflowCount();
                    }
                    break;
                } else {
                    events.add(event);
                }
            }
            if (!events.isEmpty()) {
                if (events.size() == this.batchSize) {
                    sinkCounter.incrementBatchCompleteCount();
                }
                else {
                    sinkCounter.incrementBatchUnderflowCount();
                    status = Status.BACKOFF;
                }
                // save to Hbase
                serializer.upsertEvents(events);
                sinkCounter.addToEventDrainSuccessCount(events.size());
            }
            else {
                logger.debug("no events to process ");
                sinkCounter.incrementBatchEmptyCount();
                status = Status.BACKOFF;
            }
            transaction.commit();
        } catch (ChannelException e) {
            transaction.rollback();
            status = Status.BACKOFF;
            sinkCounter.incrementConnectionFailedCount();
        }
        catch (SQLException e) {
            sinkCounter.incrementConnectionFailedCount();
            transaction.rollback();
            logger.error("exception while persisting to Hbase ", e);
            throw new EventDeliveryException("Failed to persist message to Hbase", e);
        }
        catch (Throwable e) {
            transaction.rollback();
            logger.error("exception while processing in Phoenix Sink", e);
            throw new EventDeliveryException("Failed to persist message", e);
        }
        finally {
            logger.info(String.format("Time taken to process [%s] events was [%s] seconds",
                    events.size(),
                    TimeUnit.SECONDS.convert(System.nanoTime() - startTime, TimeUnit.NANOSECONDS)));
            if( transaction != null ) {
                transaction.close();
            }
        }
        return status;
    }

    private Channel getChannel() {
        return null;
    }

    // org.apache.phoenix.mapreduce.AbstractBulkLoadTool.parseOptions(java.lang.String[])
    /**
     * Parses the commandline arguments, throws IllegalStateException if mandatory arguments are
     * missing.
     *
     * @param args supplied command line arguments
     * @return the parsed command line
     */
    //SNIPPET_STARTS
    protected CommandLine parseOptions(String[] args) {

        Options options = getOptions();

        CommandLineParser parser = new PosixParser();
        CommandLine cmdLine = null;
        try {
            cmdLine = parser.parse(options, args);
        } catch (ParseException e) {
            printHelpAndExit("Error parsing command line options: " + e.getMessage(), options);
        }

        if (cmdLine.hasOption(HELP_OPT.getOpt())) {
            printHelpAndExit(options, 0);
        }

        if (!cmdLine.hasOption(TABLE_NAME_OPT.getOpt())) {
            throw new IllegalStateException(TABLE_NAME_OPT.getLongOpt() + " is a mandatory " +
                    "parameter");
        }

        if (!cmdLine.getArgList().isEmpty()) {
            throw new IllegalStateException("Got unexpected extra parameters: "
                    + cmdLine.getArgList());
        }

        if (!cmdLine.hasOption(INPUT_PATH_OPT.getOpt())) {
            throw new IllegalStateException(INPUT_PATH_OPT.getLongOpt() + " is a mandatory " +
                    "parameter");
        }

        return cmdLine;
    }
    //SNIPPETS_END

    private void doConfigure(Context context) {

    }

    private void printHelpAndExit(Options options, int i) {

    }

    private void printHelpAndExit(String s, Options options) {

    }

    private Options getOptions() {
        return null;
    }

    private class MetaDataMutationResult {
        public MetaDataMutationResult(Object schemaNotFound, Object currentTimeMillis, Object o) {

        }
    }

    private static class MutationCode {
        public static final Object SCHEMA_NOT_FOUND = null;
        public static final Object TABLES_EXIST_ON_SCHEMA = null;
        public static final Object SCHEMA_ALREADY_EXISTS = null;
    }

    private class ImmutableBytesPtr {
        public ImmutableBytesPtr(byte[] key) {
        }
    }

    private static class EnvironmentEdgeManager {
        public static Object currentTimeMillis() {
            return null;
        }
    }

    private class PSchema {
        public long getTimeStamp() {
            return 1;
        }
    }

    private class Tuple {
    }

    private class Expression {
        public Object getSortOrder() {
            return null;
        }

        public boolean evaluate(Tuple tuple, ImmutableBytesWritable ptr) {
            return false;
        }

        public Expression getCodec() {
            return null;
        }

        public Expression getDataType() {
            return null;
        }

        public void encodeLong(long remainder, byte[] resultPtr, int i) {

        }

        public long decodeLong(ImmutableBytesWritable ptr, Object sortOrder) {
            return 0;
        }
    }

    private class RegionScanner implements AutoCloseable {
        public void next(List<Cell> results) {

        }

        @Override
        public void close() throws Exception {

        }
    }

    private class Region {
        public RegionScanner getScanner(Scan scan) {
            return null;
        }
    }

    private static class Dummy {
        public static final Dummy INSTANCE = null;

        public Region getRegion() {
            return null;
        }

        public int getByteSize() {
            return 0;
        }

        public void incrementBatchEmptyCount() throws SQLException{

        }

        public void incrementBatchUnderflowCount() {

        }

        public void incrementBatchCompleteCount() {

        }

        public void addToEventDrainSuccessCount(int size) {

        }

        public void upsertEvents(List<Event> events) {

        }

        public void incrementConnectionFailedCount() {

        }
    }

    private static class SchemaUtil {
        public static Object getKeyForSchema(Object o, String schemaName) {
            return null;
        }
    }

    private static class MetaDataUtil {
        public static Scan newTableRowsScan(Object keyForSchema, Object minTableTimestamp, long clientTimeStamp) {
            return null;
        }
    }

    private static class FlumeConstants {
        public static final Object CONFIG_TABLE_DDL = null;
        public static final Object CONFIG_TABLE = null;
        public static final Object CONFIG_ZK_QUORUM = null;
        public static final Object DEFAULT_BATCH_SIZE = null;
        public static final Object CONFIG_BATCHSIZE = null;
        public static final Object CONFIG_JDBC_URL = null;
    }

    private class Context {
        public String getString(Object configTableDdl) {
            return null;
        }

        public int getInteger(Object configBatchsize, Object defaultBatchSize) {
            return 0;
        }
    }

    private static class QueryUtil {
        public static Object getUrl(String zookeeperQuorum) {
            return null;
        }
    }

    private static class DefaultKeyGenerator {
        public static Object valueOf(String toUpperCase) {
            return null;
        }

        public static Object values() {
            return null;
        }
    }

    private class EventDeliveryException extends Exception {
        public EventDeliveryException(String s, SQLException e) {

        }

        public EventDeliveryException(String failed_to_persist_message, Throwable e) {

        }
    }

    private static class Status {
        public static final Status READY = null;
        public static final Status BACKOFF = null;
    }

    private static class TimeUnit {
        public static final Object NANOSECONDS = null;
        public static final HibernateORM.Environment SECONDS = null;
    }

    private class Transaction {
        public void begin() {

        }

        public void commit() {

        }

        public void rollback() {

        }

        public void close() {

        }
    }

    private class Event {
    }

    private class Channel {
        public Transaction getTransaction() {
            return null;
        }

        public Event take() {
            return null;
        }
    }
}
