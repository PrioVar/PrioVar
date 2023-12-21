import React, { useEffect } from 'react';
import './Team.css'; // Import the CSS file for styling

const Reports = () => {
  // Sample project data
  useEffect(() => {
    document.title = 'Reports';
  }, []);

  return (
    <div>
      <h2>Download Reports</h2>
      <ul>
        <li>
          <a
            href={process.env.PUBLIC_URL + '/T2309_Project_Specification_Document.pdf'}
            download="Project Specification Report.pdf"
          >
            Project Specification Report
          </a>
        </li>
        <li>
          <a
            href={process.env.PUBLIC_URL + '/T2309_Analysis_and_Requirements_Report.pdf'}
            download="Analysis and Requirements Report.pdf"
          >
            Analysis and Requirements Report
          </a>
        </li>
      </ul>
    </div>
  );
};

export default Reports;


/*
import React, { useEffect } from 'react';
import './Team.css'; // Import the CSS file for styling

const Reports = () => {
  // Sample project data
  useEffect(() => {
    document.title = 'Reports'
  }, []);
  
  return (
    <div>
      <p>Coming Soon...</p>
    </div>
  );
};

export default Reports;
*/