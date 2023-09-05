// SNIPPET_END_1
// S2_1:2 method chains, bad comments
/**
 * Apply the transmitted notifications (“nStore”) to the project so that acknowledged
 * notifications are deleted and other ones added.
 */
// SNIPPET_STARTS_2
public void apply212(Project project, NotificationStore nStore) {
    ProjectSpace projectSpace = project.eContainer();
    /* If projectSpace is not null */
    if (projectSpace != null) {
        /* Define local variables. */
        PropertyManager manager = projectSpace.getPropertyManager();
        StoreProperty property = manager.getLocalProperty(NOTIFICATION_COMPOSITE);
        if (property != null) {
            Value value = property.getValue();
            /* If value is not null and if nstore is acknowledged, */
            /* then remove nstore-notifications from notifications. */
            /* Otherwise, add nstore-notifications to notifications. */
            if (value != null && value instanceof NotificationComposite) {
                NotificationComposite nComposite = value;
                if (nStore.isAcknowledged()) {
                    nComposite.getNotifications().removeAll(nStore.getNotifications());
                } else {
                    nComposite.getNotifications().addAll(nStore.getNotifications());
                }
            }
        } else {
            /* If property is not null */
            /* and if nStore is not acknowledged, */
            /* then add nStore-notifications, */
            /* and set NOTIFICATION_COMPOSITE property. */
            if (!nStore.isAcknowledged()) {
                NotificationComposite nComposite = Factory.createNotificationComposite();
                nComposite.getNotifications().addAll(nStore.getNotifications());
                manager.setLocalProperty(NOTIFICATION_COMPOSITE, nComposite);
            }
        }
    }
}