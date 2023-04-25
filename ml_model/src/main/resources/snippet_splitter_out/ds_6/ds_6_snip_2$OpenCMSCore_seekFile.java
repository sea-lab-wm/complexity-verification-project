package FeatureExtraction.snippet_splitter_out;
public class ds_6_snip_2$OpenCMSCore_seekFile {
// org.opencms.jlan.CmsJlanNetworkFile.seekFile(long,int)
// @Override // Removed to allow compilation
// SNIPPET_STARTS
public long seekFile(long pos, int typ) throws IOException {
    try {
        load(true);
        switch(typ) {
            // From current position
            case SeekType.CurrentPos:
                m_buffer.seek(m_buffer.getPosition() + pos);
                break;
            // From end of file
            case SeekType.EndOfFile:
                long newPos = m_buffer.getLength() + pos;
                m_buffer.seek(newPos);
                break;
            // From start of file
            case SeekType.StartOfFile:
            default:
                m_buffer.seek(pos);
                break;
        }
        return m_buffer.getPosition();
    } catch (CmsException e) {
        throw new IOException(e);
    }
}
}