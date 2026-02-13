import java.util.Set;
import java.util.HashSet;
import java.net.URI;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;

// ADDED BY NADEESHAN
@interface Nullable {}

public class Study {

    //SNIPPET_STARTS
    /*************   Method 1   *************/ 
    public static boolean isValidProjectName(String name) {
        if (name == null) {
            return false;
        }
        if (name.startsWith(".")) {
            return false;
        }
        if ((name.length() < 1) || (name.length() > MAX_NAME_LENGTH)) {
            return false;
        }
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (!Character.isLetterOrDigit(c) && !VALID_NAME_SET.contains(c)) {
                return false;
            }
        }
        return true;
    }

    //SNIPPET_STARTS
    /*************   Method 2   *************/ 
    public static boolean isRemote(URI uri) {
        if (isFilesystemPath(uri)) {
            return false;
        }
        String scheme = uri.getScheme();
        if (scheme == null) {
            return false;
        }
        switch (scheme) {
            case "file":
            case "jar":
                return false;
            default:
                break;
        }
        return true;
    }

    //SNIPPET_STARTS
    /*************   Method 3   *************/ 
    public static boolean isMachineTypeDefined(short type) {
        if (type == IMAGE_FILE_MACHINE_UNKNOWN) {
            // Unsupported machine type
            return false;
        }
        for (Field field : CoffMachineType.class.getDeclaredFields()) {
            if (!field.isSynthetic()) {
                int modifiers = field.getModifiers();
                if (Modifier.isFinal(modifiers) && Modifier.isStatic(modifiers)) {
                    try {
                        if (field.getShort(null) == type) {
                            return true;
                        }
                    } catch (IllegalAccessException e) {
                        continue;
                    }
                }
            }
        }
        return false;
    }

    //SNIPPET_STARTS
    /*************   Method 4   *************/ 
    public static int indexOfIgnoreCase(CharSequence str, CharSequence searchStr, int startPos) {
        if (str == null || searchStr == null) {
            return INDEX_NOT_FOUND;
        }
        if (startPos < 0) {
            startPos = 0;
        }
        int searchStrLen = searchStr.length();
        int endLimit = str.length() - searchStrLen + 1;
        if (startPos > endLimit) {
            return INDEX_NOT_FOUND;
        }
        if (searchStrLen == 0) {
            return startPos;
        }
        for (int i = startPos; i < endLimit; i++) {
            if (regionMatches(str, true, i, searchStr, 0, searchStrLen)) {
                return i;
            }
        }
        return INDEX_NOT_FOUND;
    }

    //SNIPPET_STARTS
    /*************   Method 5   *************/ 
    public static boolean deleteRecursively(@Nullable Path root) throws IOException {
        if (root == null) return false;
        if (!Files.exists(root)) return false;

        Files.walkFileTree(root, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                Files.delete(file);
                return FileVisitResult.CONTINUE;
            }
            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                Files.delete(dir);
                return FileVisitResult.CONTINUE;
            }
        });

        return true;
    }

    //SNIPPET_STARTS
    /*************   Method 6   *************/ 
    public static int encodedLength(CharSequence sequence) {
        // Optimized implementation
        int utf16Length = sequence.length();
        int utf8Length = utf16Length;
        int i = 0;

        while (i < utf16Length && sequence.charAt(i) < 0x80) {
            i++;
        }

        for (; i < utf16Length; i++) {
            char c = sequence.charAt(i);
            if (c < 0x800) {
                utf8Length += ((0x7f - c) >>> 31);
            } else {
                utf8Length += encodedLengthGeneral(sequence, i);
                break;
            }
        }

        if (utf8Length < utf16Length) {
            throw new IllegalArgumentException(
                "UTF-8 length does not fit in int: " + (utf8Length + (1L << 32)));
        }

        return utf8Length;
    }

    //SNIPPET_STARTS
    /*************   Method 7   *************/ 
    public static float lowestPositiveRoot(float a, float b, float c) {
        float det = b * b - 4 * a * c;
        if (det < 0) return Float.NaN;

        float sqrtD = (float)Math.sqrt(det);
        float invA = 1 / (2 * a);
        float r1 = (-b - sqrtD) * invA;
        float r2 = (-b + sqrtD) * invA;

        if (r1 > r2) {
            float tmp = r2;
            r2 = r1;
            r1 = tmp;
        }

        if (r1 > 0) return r1;
        if (r2 > 0) return r2;
        return Float.NaN;
    }

    //SNIPPET_STARTS
    /*************   Method 8   *************/ 
    public static float atan2(float y, float x) {
        float n = y / x;

        if (n != n)
            n = (y == x ? 1f : -1f);
        else if (n - n != n - n)
            x = 0f;

        if (x > 0)
            return atanUnchecked(n);
        else if (x < 0) {
            if (y >= 0) return atanUnchecked(n) + PI;
            return atanUnchecked(n) - PI;
        } else if (y > 0)
            return x + HALF_PI;
        else if (y < 0)
            return x - HALF_PI;

        return x + y;
    }
    //SNIPPETS_END

    // ADDED BY NADEESHAN
    private static final int MAX_NAME_LENGTH = 255;

    private static final int INDEX_NOT_FOUND = -1;

    private static final Set<Character> VALID_NAME_SET = new HashSet<>();

    static {
        VALID_NAME_SET.add('_');
        VALID_NAME_SET.add('-');
    }


    private static boolean regionMatches(
            CharSequence cs,
            boolean ignoreCase,
            int thisStart,
            CharSequence substring,
            int start,
            int length) {
        return false;
    }

    private static boolean isFilesystemPath(URI uri) {
        if (uri == null) {
            return false;
        }
        return uri.getScheme() == null;
    }

    private static final short IMAGE_FILE_MACHINE_UNKNOWN = 0;

    private static final float PI = 3.1415927f;
    private static final float HALF_PI = PI / 2f;

    private static class CoffMachineType {
        public static final short IMAGE_FILE_MACHINE_I386  = (short) 0x014c;
        public static final short IMAGE_FILE_MACHINE_AMD64 = (short) 0x8664;
        public static final short IMAGE_FILE_MACHINE_ARM64 = (short) 0xAA64;
    }

    private static int encodedLengthGeneral(CharSequence sequence, int start) {
        return 0;
    }

    private static float atanUnchecked(float x) {
        return 0f;
    }
}