package snippet_splitter_out.ds_9$gc;
public class ds_9$gc_snip_4_apply {
// SNIPPET_END_3
// S2_2:1 resolved method chains, good comments
/**
 * Apply the transmitted notifications (“nStore”) to the project so that
 * acknowledged notifications are deleted and other ones added.
 */
// SNIPPET_STARTS_1
public void apply(Project project, NotificationStore nStore) {
    ProjectSpace projectSpace = project.eContainer();
    /* If project space has not been initialized, there is nothing to do. */
    if (projectSpace != null) {
        /* Get project properties and check if there are notifications. */
        PropertyManager manager = projectSpace.getPropertyManager();
        StoreProperty property = manager.getLocalProperty(NOTIFICATION_COMPOSITE);
        NotificationList notifications;
        if (property != null) {
            Value value = property.getValue();
            /* If the project already has notifications */
            /* and if transmitted notifications are acknowledged, */
            /* then remove the transmitted notifications from the project. */
            /* Otherwise, add the transmitted notifications to the project. */
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
            /* If the project did not have notifications yet */
            /* and if transmitted notifications are not acknowledged, */
            /* then add the transmitted notifications to the project */
            /* and store them in the NOTIFICATION_COMPOSITE property. */
            if (!nStore.isAcknowledged()) {
                NotificationComposite nComposite = Factory.createNotificationComposite();
                notifications = nComposite.getNotifications();
                notifications.addAll(nStore.getNotifications());
                manager.setLocalProperty(NOTIFICATION_COMPOSITE, nComposite);
            }
        }
    }
}
}