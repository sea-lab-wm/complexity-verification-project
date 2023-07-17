package snippet_splitter_out.ds_6;

public class ds_6_snip_5$K9_testTextQuoteToHtmlBlockquote {
  // com.fsck.k9.message.html.HtmlConverterTest.testTextQuoteToHtmlBlockquote()
  // @Test // Removed to allow compilation
  // SNIPPET_STARTS
  public void testTextQuoteToHtmlBlockquote() {
    String message =
        "Panama!\r\n"
            + "\r\n"
            + "Bob Barker <bob@aol.com> wrote:\r\n"
            + "> a canal\r\n"
            + ">\r\n"
            + "> Dorothy Jo Gideon <dorothy@aol.com> espoused:\r\n"
            + "> >A man, a plan...\r\n"
            + "> Too easy!\r\n"
            + "\r\n"
            + "Nice job :)\r\n"
            + ">> Guess!";
    String result = HtmlConverter.textToHtml(message);
    writeToFile(result);
    assertEquals(
        "<pre class=\"k9mail\">"
            + "Panama!<br />"
            + "<br />"
            + "Bob Barker &lt;bob@aol.com&gt; wrote:<br />"
            + "<blockquote class=\"gmail_quote\" style=\"margin: 0pt 0pt 1ex 0.8ex; border-left: 1px solid #729fcf; padding-left: 1ex;\">"
            + " a canal<br />"
            + "<br />"
            + " Dorothy Jo Gideon &lt;dorothy@aol.com&gt; espoused:<br />"
            + "<blockquote class=\"gmail_quote\" style=\"margin: 0pt 0pt 1ex 0.8ex; border-left: 1px solid #ad7fa8; padding-left: 1ex;\">"
            + "A man, a plan...<br />"
            + "</blockquote>"
            + " Too easy!<br />"
            + "</blockquote>"
            + "<br />"
            + "Nice job :)<br />"
            + "<blockquote class=\"gmail_quote\" style=\"margin: 0pt 0pt 1ex 0.8ex; border-left: 1px solid #729fcf; padding-left: 1ex;\">"
            + "<blockquote class=\"gmail_quote\" style=\"margin: 0pt 0pt 1ex 0.8ex; border-left: 1px solid #ad7fa8; padding-left: 1ex;\">"
            + " Guess!"
            + "</blockquote>"
            + "</blockquote>"
            + "</pre>",
        result);
  }
}
