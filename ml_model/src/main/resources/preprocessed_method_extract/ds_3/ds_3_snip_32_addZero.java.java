// Added to allow compilation
// Snippet s67
/**
 * Add one zero if neccessary
 *
 * @param number
 * @return
 */
// SNIPPET_STARTS
private CharSequence addZero(int number) {
    StringBuilder builder = new StringBuilder();
    if (number < 10) {
        builder.append('0');
    }
    builder.append(Integer.toString(number));
    return builder;
    /*Altered return*/
    // return null; // Added to allow compilation
}