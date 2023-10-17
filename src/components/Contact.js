import React, { useEffect } from 'react';
import { Mail } from '@mui/icons-material';

const ContactMe = () => {
  useEffect(() => {
    document.title = 'Contact PrioVar'
    
  }, []);
  return (
    <div>
      <h2>Contact PrioVar!</h2>
      <p>You can contact us via following channels:</p>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        <li style={{ marginBottom: '0.5rem' }}>
          <Mail style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
          <a href="erkin_mail" style={{ textDecoration: 'none' }}>erkin.aydin@ug.bilkent.edu.tr</a>
          <br/>
          <Mail style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
          <a href="alperen_mail" style={{ textDecoration: 'none' }}>alperen.gozeten@ug.bilkent.edu.tr</a>
        </li>
      </ul>
    </div>
  );
};

export default ContactMe;