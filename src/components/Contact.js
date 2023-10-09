import React, { useEffect } from 'react';
import { Mail } from '@mui/icons-material';

const ContactMe = () => {
  useEffect(() => {
    document.title = 'Contact PrioVar'
    
  }, []);
  return (
    <div>
      <h2>Contact Xaga!</h2>
      <p>You can contact us via following channels:</p>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        <li style={{ marginBottom: '0.5rem' }}>
          <Mail style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
          <a href="erkinresearch@gmail.com" style={{ textDecoration: 'none' }}>priovarmail@dummy.com</a>
        </li>
      </ul>
    </div>
  );
};

export default ContactMe;