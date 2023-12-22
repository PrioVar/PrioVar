// Reports.jsx

import React, { useEffect, useState } from 'react';
import { InsertDriveFile as InsertDriveFileIcon } from '@mui/icons-material';
//import FileDownload from 'react-file-download';
import './Team.css'; // Import the CSS file for styling

const Reports = () => {
  // Sample project data
  useEffect(() => {
    document.title = 'Reports';
  }, []);

  /*
  const handleDownload = (filename) => {
    const fileUrl = process.env.PUBLIC_URL + '/' + filename;
    FileDownload(fileUrl, filename);
  };

  // Function to handle downloading reports
  const handleReportDownload = (pdfUrl, filename) => {
    const link = document.createElement("a");
    link.href = pdfUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Functions for downloading specific reports
  const downloadSpecification = () => {
    handleReportDownload("T2309_Project_Specification_Document.pdf", "T2309_Project_Specification_Document.pdf");
  };

  const downloadAnalysis = () => {
    handleReportDownload("T2309_Analysis_and_Requirements_Report.pdf", "T2309_Analysis_and_Requirements_Report.pdf");
  };

  const downloadDemoPresentation = () => {
    handleReportDownload("PrioVar_491_Demo_Presentation.pdf", "PrioVar_491_Demo_Presentation.pdf");
  };
  */
  
  const [showSpecification, setShowSpecification] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [showDemoPresentation, setShowDemoPresentation] = useState(false);

  const toggleVisibility = (reportType) => {
    switch (reportType) {
      case 'specification':
        setShowSpecification(!showSpecification);
        setShowAnalysis(false);
        setShowDemoPresentation(false);
        break;
      case 'analysis':
        setShowAnalysis(!showAnalysis);
        setShowSpecification(false);
        setShowDemoPresentation(false);
        break;
      case 'demoPresentation':
        setShowDemoPresentation(!showDemoPresentation);
        setShowSpecification(false);
        setShowAnalysis(false);
        break;
      default:
        break;
    }
  };

  return (
    <div className="reports-container">
      <div className="button-section">
        <div className="report-section">
          <h2>Reports</h2>
          <ul className="button-list">
            <li>
              <button onClick={() => toggleVisibility('specification')}>
                <InsertDriveFileIcon className="pdf-icon" />
                Project Specification Report
              </button>
            </li>
            <li>
              <button onClick={() => toggleVisibility('analysis')}>
                <InsertDriveFileIcon className="pdf-icon" />
                Analysis and Requirements Report
              </button>
            </li>
          </ul>
        </div>
        <div className="presentation-section">
          <h2>Presentations</h2>
          <ul className="button-list">
            <li>
              <button onClick={() => toggleVisibility('demoPresentation')}>
                <InsertDriveFileIcon className="pdf-icon" />
                Demo Presentation
              </button>
            </li>
          </ul>
        </div>
      </div>
      <div className="preview-section">
        {showSpecification && (
          <iframe title="Project Specification Report" src="T2309_Project_Specification_Document.pdf" width="250%" height="450px"></iframe>
        )}
        {showAnalysis && (
          <iframe title="Analysis and Requirements Report" src="T2309_Analysis_and_Requirements_Report.pdf" width="250%" height="450px"></iframe>
        )}
        {showDemoPresentation && (
          <iframe title="Demo Presentation" src="PrioVar_491_Demo_Presentation.pdf" width="250%" height="300px"></iframe>
        )}
      </div>
    </div>
  );
};

export default Reports;
