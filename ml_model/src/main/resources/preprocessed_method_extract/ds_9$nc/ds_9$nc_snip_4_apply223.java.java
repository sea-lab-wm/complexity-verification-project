// SNIPPET_END_2
// S2_2:3 resolved method chains, no comments
/**
 * Apply the transmitted notifications (“nStore”) to the project so that acknowledged
 * notifications are deleted and other ones added.
 */
// SNIPPET_STARTS_3
public void apply223(Project project, NotificationStore nStore) {
    ProjectSpace projectSpace = project.eContainer();
    if (projectSpace != null) {
        PropertyManager manager = projectSpace.getPropertyManager();
        StoreProperty property = manager.getLocalProperty(NOTIFICATION_COMPOSITE);
        NotificationList notifications;
        if (property != null) {
            Value value = property.getValue();
            if (value != null && value instanceof NotificationComposite) {
                NotificationComposite nComposite = value;
                notifications = nComposite.getNotifications();
                if (nStore.isAcknowledged()) {
                    notifications.removeAll(nStore.getNotifications());
                } else {
                    notifications.addAll(nStore.getNotifications());
                }
            }
        } else {
            if (!nStore.isAcknowledged()) {
                NotificationComposite nComposite = Factory.createNotificationComposite();
                notifications = nComposite.getNotifications();
                notifications.addAll(nStore.getNotifications());
                manager.setLocalProperty(NOTIFICATION_COMPOSITE, nComposite);
            }
        }
    }
}