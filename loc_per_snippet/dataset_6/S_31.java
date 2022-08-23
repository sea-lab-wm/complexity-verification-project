    //SNIPPET_STARTS
    public long seekFile(long pos, int typ) throws IOException {

        try {
            load(true);
            switch (typ) {

                //  From current position

                case SeekType.CurrentPos:
                    m_buffer.seek(m_buffer.getPosition() + pos);
                    break;

                //  From end of file

                case SeekType.EndOfFile:
                    long newPos = m_buffer.getLength() + pos;
                    m_buffer.seek(newPos);
                    break;

                //  From start of file

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

    // org.opencms.main.CmsShell.execute(java.io.Reader)
    /**
     * Executes the commands from the given reader in this shell.<p>
     *
     * <ul>
     * <li>Commands in the must be separated with a line break '\n'.
     * <li>Only one command per line is allowed.
     * <li>String parameters must be quoted like this: <code>'string value'</code>.
     * </ul>
     *
     * @param reader the reader from which the commands are read
     */
