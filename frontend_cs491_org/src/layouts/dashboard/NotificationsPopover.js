import React, { useRef, useState, useEffect } from 'react';
import { Icon } from '@iconify/react';
import bellFill from '@iconify/icons-eva/bell-fill';
import clockFill from '@iconify/icons-eva/clock-fill';
import doneAllFill from '@iconify/icons-eva/done-all-fill'; // Icon for "mark all as read"
import {
  alpha, Box, List, Badge, Button, Avatar, Tooltip, Divider, Typography,
  ListItemText, ListItemButton
} from '@material-ui/core';
import Scrollbar from '../../components/Scrollbar';
import MenuPopover from '../../components/MenuPopover';
import { MIconButton } from '../../components/@material-extend';
import NotificationDetailsDialog from './NotificationDetailsDialog';
import { markNotificationAsRead, markAllNotificationsAsRead, fetchNotifications, acceptInformationRequest, rejectInformationRequest } from '../../api/file';  // Ensure this points to your actual import path
import { useSnackbar } from 'notistack5'
import closeFill from '@iconify/icons-eva/close-fill'

function NotificationsPopover() {
  const anchorRef = useRef(null);
  const [open, setOpen] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [selectedNotification, setSelectedNotification] = useState(null);
  const [responseMessage, setResponseMessage] = useState('');
  const { enqueueSnackbar, closeSnackbar } = useSnackbar()

  const actorId = parseInt(localStorage.getItem('clinicianId') === '-1' ? localStorage.getItem('healthCenterId') : localStorage.getItem('clinicianId'));

  useEffect(() => {
    fetchNotifications(actorId)
      .then(data => {
        // Sort notifications by sendAt in descending order
        data.sort((a, b) => new Date(b.sendAt) - new Date(a.sendAt));
        setNotifications(data);
      })
      .catch(error => console.error('Failed to load notifications:', error));
  }, []);

  const handleOpen = () => setOpen(true);
  const handleClose = () => {
    setOpen(false);
    setSelectedNotification(null);
    setResponseMessage('');
  };
  const handleAccept = async () => {

    try {
      handleClose();
      enqueueSnackbar('Response sent successfully!', {
        variant: 'success',
        action: (key) => (
          <MIconButton size="small" onClick={() => closeSnackbar(key)}>
            <Icon icon={closeFill} />
          </MIconButton>
        ),
      })
      if (selectedNotification && selectedNotification.informationRequest) {
        await acceptInformationRequest(selectedNotification.informationRequest.id, responseMessage);
      }
    } catch (error) {
      handleClose();
      enqueueSnackbar("Failed to send request", {
        variant: 'error',
        action: (key) => (
          <MIconButton size="small" onClick={() => closeSnackbar(key)}>
            <Icon icon={closeFill} />
          </MIconButton>
        ),
      })
    }
    
  };

  const handleReject = async () => {
    try {
      handleClose();
      alert("Response sent successfully");
      if (selectedNotification && selectedNotification.informationRequest) {
        await rejectInformationRequest(selectedNotification.informationRequest.id, responseMessage);
      }
    } catch (error) {
      handleClose();
      alert("Failed to send request");
    }
  };


  const handleNotificationClick = async (notification) => {
    if (!notification.isRead) {
      try {
        await markNotificationAsRead(notification.id);
        setNotifications(notifications.map(n => n.id === notification.id ? { ...n, isRead: true } : n));
      } catch (error) {
        console.error('Error marking notification as read:', error);
      }
    }
    setSelectedNotification(notification);
  };

  const handleMarkAllAsRead = async () => {
    try {
      await markAllNotificationsAsRead(actorId);
      setNotifications(notifications.map(n => ({ ...n, isRead: true })));
    } catch (error) {
      console.error('Error marking all notifications as read:', error);
    }
  };

  return (
    <>
      <MIconButton
        ref={anchorRef}
        size="large"
        color={open ? 'primary' : 'default'}
        onClick={handleOpen}
        sx={{
          ...(open && {
            bgcolor: (theme) => alpha(theme.palette.primary.main, theme.palette.action.focusOpacity),
          }),
        }}
      >
        <Badge badgeContent={notifications.filter(n => !n.isRead).length} color="error">
          <Icon icon={bellFill} width={20} height={20} />
        </Badge>
      </MIconButton>

      <MenuPopover open={open} onClose={handleClose} anchorEl={anchorRef.current} sx={{ width: 360 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', py: 2, px: 2.5 }}>
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="subtitle1">Notifications</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              You have {notifications.filter(n => !n.isRead).length} unread notifications
            </Typography>
          </Box>
          <Tooltip title="Mark all as read">
            <MIconButton color="primary" onClick={handleMarkAllAsRead}>
              <Icon icon={doneAllFill} width={20} height={20} />
            </MIconButton>
          </Tooltip>
        </Box>

        <Divider />

        <Scrollbar sx={{ height: 340 }}>
          <List disablePadding>
            {notifications.map((notification) => (
              <ListItemButton key={notification.id} onClick={() => handleNotificationClick(notification)} disableGutters sx={{
                py: 1.5, px: 2.5, mt: '1px',
                bgcolor: notification.isRead ? '#C8E6C9' : 'inherit',  // Applying green background if read
              }}>
                <Avatar sx={{ bgcolor: 'background.neutral' }}>{/* Placeholder for notification icon */}</Avatar>
                <ListItemText
                  primary={notification.notification}
                  secondary={
                    <Typography variant="caption" sx={{ mt: 0.5, display: 'flex', alignItems: 'center', color: 'text.disabled' }}>
                      <Box component={Icon} icon={clockFill} sx={{ mr: 0.5, width: 16, height: 16 }} />
                      {new Intl.DateTimeFormat('default', {
                        year: 'numeric', month: 'long', day: '2-digit',
                        hour: '2-digit', minute: '2-digit'
                      }).format(new Date(notification.sendAt))}
                    </Typography>
                  }
                />
              </ListItemButton>
            ))}
          </List>
        </Scrollbar>

        <Divider />
      </MenuPopover>
      <NotificationDetailsDialog
        selectedNotification={selectedNotification}
        responseMessage={responseMessage}
        setResponseMessage={setResponseMessage}
        handleClose={handleClose}
        handleAccept={handleAccept}
        handleReject={handleReject}
      />
    </>
  );
}

export default NotificationsPopover;
