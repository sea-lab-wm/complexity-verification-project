    //SNIPPET_STARTS
    public Column[] getOverriddenColumn(String propertyName) {
        Column[] result = getExactOverriddenColumn( propertyName );
        if (result == null) {
            //the commented code can be useful if people use the new prefixes on old mappings and vice versa
            // if we enable them:
            // WARNING: this can conflict with user's expectations if:
            //  - the property uses some restricted values
            //  - the user has overridden the column
            // also change getOverriddenJoinColumn and getOverriddenJoinTable as well

//			if ( propertyName.contains( ".key." ) ) {
//				//support for legacy @AttributeOverride declarations
//				//TODO cache the underlying regexp
//				result = getExactOverriddenColumn( propertyName.replace( ".key.", ".index."  ) );
//			}
//			if ( result == null && propertyName.endsWith( ".key" ) ) {
//				//support for legacy @AttributeOverride declarations
//				//TODO cache the underlying regexp
//				result = getExactOverriddenColumn(
//						propertyName.substring( 0, propertyName.length() - ".key".length() ) + ".index"
//						);
//			}
//			if ( result == null && propertyName.contains( ".value." ) ) {
//				//support for legacy @AttributeOverride declarations
//				//TODO cache the underlying regexp
//				result = getExactOverriddenColumn( propertyName.replace( ".value.", ".element."  ) );
//			}
//			if ( result == null && propertyName.endsWith( ".value" ) ) {
//				//support for legacy @AttributeOverride declarations
//				//TODO cache the underlying regexp
//				result = getExactOverriddenColumn(
//						propertyName.substring( 0, propertyName.length() - ".value".length() ) + ".element"
//						);
//			}
            if ( result == null && propertyName.contains( ".collection&&element." ) ) {
                //support for non map collections where no prefix is needed
                //TODO cache the underlying regexp
                result = getExactOverriddenColumn( propertyName.replace( ".collection&&element.", "."  ) );
            }
        }
        return result;
    }

    private static class CollectionPersister {
        public char[] getRole() {
            return new char[0];
        }

        public CollectionPersister getOwnerEntityPersister() {
            return null;
        }

        public Type getIdentifierType() {
            return null;
        }
    }

    private static class PersistentCollection {
        public Object getOwner() {
            return null;
        }
    }

    private static class Type {
        public Class<?> getReturnedClass() {
            return null;
        }

        public char[] toLoggableString(Serializable ownerKey, Object factory) {
            return new char[0];
        }
    }

    private static class EntityEntry {
        public Serializable getId() {
            return null;
        }
    }

