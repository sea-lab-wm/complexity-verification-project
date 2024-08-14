package snippet_splitter_out.ds_6;
public class ds_6_snip_4$SpringBatch_testCustomRecordSeparatorMultilineBlankLineAfterEnd {
// org.springframework.batch.item.file.FlatFileItemReaderTests.testCustomRecordSeparatorMultilineBlankLineAfterEnd()
// @Test // removed to allow compilation
// SNIPPET_STARTS
public void testCustomRecordSeparatorMultilineBlankLineAfterEnd() throws Exception {
    reader.setRecordSeparatorPolicy(new RecordSeparatorPolicy() {

        // 1 record = 2 lines
        boolean pair = true;

        @Override
        public boolean isEndOfRecord(String line) {
            if (StringUtils.hasText(line)) {
                pair = !pair;
            }
            return pair;
        }

        @Override
        public String postProcess(String record) {
            return StringUtils.hasText(record) ? record : null;
        }

        @Override
        public String preProcess(String record) {
            return record;
        }
    });
    reader.setResource(getInputResource("testLine1\ntestLine2\n\n"));
    reader.open(executionContext);
    assertEquals("testLine1testLine2", reader.read());
    assertEquals(null, reader.read());
}
}