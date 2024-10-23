package snippet_splitter_out.ds_6;

public class ds_6_snip_2$Phoenix_evaluate {
  // org.apache.phoenix.expression.ModulusExpression.evaluate(org.apache.phoenix.schema.tuple.Tuple,org.apache.hadoop.hbase.io.ImmutableBytesWritable)
  // @Override // removed to allow compilation
  // SNIPPET_STARTS
  public boolean evaluate(Tuple tuple, ImmutableBytesWritable ptr) {
    // get the dividend
    Expression dividendExpression = getDividendExpression();
    if (!dividendExpression.evaluate(tuple, ptr)) {
      return false;
    }
    if (ptr.getLength() == 0) {
      return true;
    }
    long dividend =
        dividendExpression
            .getDataType()
            .getCodec()
            .decodeLong(ptr, dividendExpression.getSortOrder());
    // get the divisor
    Expression divisorExpression = getDivisorExpression();
    if (!divisorExpression.evaluate(tuple, ptr)) {
      return false;
    }
    if (ptr.getLength() == 0) {
      return true;
    }
    long divisor =
        divisorExpression
            .getDataType()
            .getCodec()
            .decodeLong(ptr, divisorExpression.getSortOrder());
    // actually perform modulus
    long remainder = dividend % divisor;
    // return the result, use encodeLong to avoid extra Long allocation
    byte[] resultPtr = new byte[PLong.INSTANCE.getByteSize()];
    getDataType().getCodec().encodeLong(remainder, resultPtr, 0);
    ptr.set(resultPtr);
    return true;
  }
}
