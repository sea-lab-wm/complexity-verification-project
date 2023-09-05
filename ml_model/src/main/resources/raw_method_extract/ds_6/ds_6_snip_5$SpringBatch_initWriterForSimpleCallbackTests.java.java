// org.springframework.batch.item.xml.StaxEventItemWriterTests.initWriterForSimpleCallbackTests()
// SNIPPET_STARTS
private void initWriterForSimpleCallbackTests() throws Exception {
    writer = createItemWriter();
    writer.setHeaderCallback(new StaxWriterCallback() {

        @Override
        public void write(XMLEventWriter writer) throws IOException {
            XMLEventFactory factory = XMLEventFactory.newInstance();
            try {
                writer.add(factory.createStartElement("ns", "https://www.springframework.org/test", "group"));
            } catch (XMLStreamException e) {
                throw new RuntimeException(e);
            }
        }
    });
    writer.setFooterCallback(new StaxWriterCallback() {

        @Override
        public void write(XMLEventWriter writer) throws IOException {
            XMLEventFactory factory = XMLEventFactory.newInstance();
            try {
                writer.add(factory.createEndElement("ns", "https://www.springframework.org/test", "group"));
            } catch (XMLStreamException e) {
                throw new RuntimeException(e);
            }
        }
    });
    writer.setRootTagName("{https://www.springframework.org/test}ns:testroot");
    writer.afterPropertiesSet();
}