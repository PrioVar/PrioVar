// NotificationDetailsDialog.jsx
import React from 'react';
import {
  Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions, Button, TextField
} from '@material-ui/core';

const NotificationDetailsDialog = React.memo(({ selectedNotification, responseMessage, setResponseMessage, handleClose, handleAccept, handleReject }) => (
  <Dialog open={Boolean(selectedNotification)} onClose={handleClose}>
    <DialogTitle>Notification Details</DialogTitle>
    <DialogContent>
      <DialogContentText>
        {selectedNotification?.notification}
        <br />
        Message: {selectedNotification?.appendix}
      </DialogContentText>
      {selectedNotification?.type === 'REQUEST' && (
        <TextField
          fullWidth
          multiline
          rows={4}
          variant="outlined"
          label="Response Message"
          value={responseMessage}
          onChange={(e) => setResponseMessage(e.target.value)}
          sx={{ mt: 2 }}
        />
      )}
    </DialogContent>
    <DialogActions>
      {selectedNotification?.type === 'REQUEST' && (
        <>
          <Button type="button" onClick={handleAccept} color="primary">Accept</Button>
          <Button type="button" onClick={handleReject} color="secondary">Reject</Button>
        </>
      )}
      <Button type="button" onClick={handleClose}>Close</Button>
    </DialogActions>
  </Dialog>
));

export default NotificationDetailsDialog;
