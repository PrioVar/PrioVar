import React, { useEffect } from 'react';
import { Grid, Paper, Container } from '@mui/material';
import { GitHub } from '@mui/icons-material';
import './Team.css'; // Import the CSS file for styling
import CiceklabImage from "./ciceklab.png";
const Team = () => {
  // Sample project data
  useEffect(() => {
    document.title = 'Meet the Xaga Team'
  }, []);
  
  return (
    <div>
      <h2>Meet the Xaga Team</h2>
        <Container className="centered-container">
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#c0c2c4" }}>
                <div className="person-content" >
                  <b>Alperen Gözeten</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                    <p>Small explanation goes here</p>
                  </div>
                  <a href="https://github.com/alperengozeten" target="_blank" rel="noopener noreferrer">
                    <GitHub />
                  </a>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#c0c2c4" }}>
                <div className="person-content">
                  <b>Korhan Kemal Kaya</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                    <p>Small explanation goes here</p>
                  </div>
                  <a href="https://github.com/korhankemalkaya" target="_blank" rel="noopener noreferrer">
                    <GitHub />
                  </a>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#c0c2c4" }}>
                <div className="person-content">
                  <b>Safa Eren Kuday</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                    <br/>
                    <em>Bilkent MATH - Double Major</em>
                    <p>Small explanation goes here</p>
                  </div>
                  <a href="https://github.com/safa54" target="_blank" rel="noopener noreferrer">
                    <GitHub />
                  </a>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#c0c2c4" }}>
                <div className="person-content">
                  <b>Kaan Tek</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                    <p>Small explanation goes here</p>
                  </div>
                  <a href="https://github.com/KaanTekTr" target="_blank" rel="noopener noreferrer">
                    <GitHub />
                  </a>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#c0c2c4" }}>
                <div className="person-content">
                  <b>Erkin Aydın</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                    <p>Small explanation goes here</p>
                  </div>
                  <a href="https://github.com/Erkin-Aydin" target="_blank" rel="noopener noreferrer">
                    <GitHub />
                  </a>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#c0c2c4"}}>
                <div className="person-content">
                  <b>Abdullah Ercüment Çiçek</b>
                  <p>Small explanation goes here</p>
                  <a href="http://ciceklab.cs.bilkent.edu.tr/" target="_blank" rel="noopener noreferrer">
                    <img src={CiceklabImage} alt="Link to Çiçeklab"/>
                  </a>
                </div>
              </Paper>
            </Grid>
          </Grid>
        </Container>
    </div>
  );
};

export default Team;
