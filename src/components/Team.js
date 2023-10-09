import React, { useEffect } from 'react';
import { Grid, Paper, Container } from '@mui/material';
import { GitHub, LinkedIn } from '@mui/icons-material';
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
              <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#d3d4d5" }}>
                <div className="person-content" style={{ paddingTop: "1rem"}} >
                  <b>Alperen Gözeten</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                    <p>Small explanation goes here</p>
                  </div>
                  <span>
                    <a href="https://github.com/alperengozeten" target="_blank" rel="noopener noreferrer">
                      <GitHub style={{ fontSize: '60px' }} />
                    </a>
                    <a href="https://www.linkedin.com/in/alperen-g%C3%B6zeten-62818a209/" target="_blank" rel="noopener noreferrer">
                      <LinkedIn style={{ fontSize: '60px', marginLeft: "1rem"}} />
                    </a>
                  </span>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#d3d4d5" }}>
                <div className="person-content" style={{ paddingTop: "1rem"}}>
                  <b>Korhan Kemal Kaya</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                    <p>Small explanation goes here</p>
                  </div>
                  <span>
                    <a href="https://github.com/korhankemalkaya" target="_blank" rel="noopener noreferrer">
                      <GitHub style={{ fontSize: '60px' }} />
                    </a>
                    <a href="hhttps://www.linkedin.com/in/korhankemalkaya/" target="_blank" rel="noopener noreferrer">
                      <LinkedIn style={{ fontSize: '60px', marginLeft: "1rem"}} />
                    </a>
                  </span>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#d3d4d5" }}>
                <div className="person-content" style={{ paddingTop: "1rem"}}>
                  <b>Safa Eren Kuday</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                    <br/>
                    <em>Bilkent MATH - Double Major</em>
                    <p>Small explanation goes here</p>
                  </div>
                  <span>
                    <a href="https://github.com/safa54" target="_blank" rel="noopener noreferrer">
                      <GitHub style={{ fontSize: '60px' }} />
                    </a>
                    <a href="https://www.linkedin.com/in/safa-eren-kuday/" target="_blank" rel="noopener noreferrer">
                      <LinkedIn style={{ fontSize: '60px', marginLeft: "1rem"}} />
                    </a>
                  </span>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#d3d4d5" }}>
                <div className="person-content" style={{ paddingTop: "1rem"}}>
                  <b>Kaan Tek</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                    <p>Small explanation goes here</p>
                  </div>
                  <span>
                    <a href="https://github.com/KaanTekTr" target="_blank" rel="noopener noreferrer">
                      <GitHub style={{ fontSize: '60px' }} />
                    </a>
                    <a href="https://www.linkedin.com/in/kaan-tek-a299bb195/" target="_blank" rel="noopener noreferrer">
                      <LinkedIn style={{ fontSize: '60px', marginLeft: "1rem"}} />
                    </a>
                  </span>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#d3d4d5" }}>
                <div className="person-content" style={{ paddingTop: "1rem"}}>
                  <b>Erkin Aydın</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                    <p>Small explanation goes here</p>
                  </div>
                  <span>
                    <a href="https://github.com/Erkin-Aydin" target="_blank" rel="noopener noreferrer">
                      <GitHub style={{ fontSize: '60px' }} />
                    </a>
                    <a href="https://www.linkedin.com/in/erkin-ayd%C4%B1n-167aa8229/" target="_blank" rel="noopener noreferrer">
                      <LinkedIn style={{ fontSize: '60px', marginLeft: "1rem"}} />
                    </a>
                  </span>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#d3d4d5"}}>
                <div className="person-content" style={{ paddingTop: "1rem"}}>
                  <b>Abdullah Ercüment Çiçek</b>
                  <p>Small explanation goes here</p>
                  <span>
                    <a href="http://ciceklab.cs.bilkent.edu.tr/" target="_blank" rel="noopener noreferrer">
                      <img src={CiceklabImage} alt="Link to Çiçeklab"/>
                    </a>
                    <a href="https://github.com/Erkin-Aydin" target="_blank" rel="noopener noreferrer">
                      <GitHub style={{ fontSize: '60px', marginLeft: "2.5rem" }} />
                    </a>
                    <a href="https://www.linkedin.com/in/ercumentcicek/" target="_blank" rel="noopener noreferrer">
                      <LinkedIn style={{ fontSize: '60px', marginLeft: "1rem"}} />
                    </a>
                  </span>
                </div>
              </Paper>
            </Grid>
          </Grid>
        </Container>
    </div>
  );
};

export default Team;
