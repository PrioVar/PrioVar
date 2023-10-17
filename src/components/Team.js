import React, { useEffect } from 'react';
import { Grid, Paper, Container } from '@mui/material';
import { GitHub, LinkedIn } from '@mui/icons-material';
import './Team.css'; // Import the CSS file for styling
import CiceklabImage from "./ciceklab.png";
import Alperen from "./alperen.jpeg";
import Safa from "./safa.jpeg";
import Kaan from "./kaan.jpeg";
import ECicek from "./e-cicek.jpeg";
import Erkin from "./erkin.jpeg";
import Korhan from "./korhan.jpeg";

const Team = () => {
  // Sample project data
  useEffect(() => {
    document.title = 'Meet the PrioVar Team'
  }, []);
  
  return (
    <div>
      <h2>Meet the PrioVar Team</h2>
        <Container className="centered-container">
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Paper elevation={3} className="person-box" style={{ maxWidth: "40rem", backgroundColor: "#d3d4d5" }}>
                <div className="person-content" style={{ paddingTop: "1rem"}} >
                  <img src={Alperen} alt="alperen" style={{width: "100px", borderRadius: "50%"}}/>
                  <br/>
                  <b>Alperen Gözeten</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                  </div>
                  <span>
                    <a href="https://github.com/alperengozeten" target="_blank" rel="noopener noreferrer">
                      <GitHub style={{ fontSize: '60px', color: '#555'}} />
                    </a>
                    <a href="https://www.linkedin.com/in/alperen-g%C3%B6zeten-62818a209/" target="_blank" rel="noopener noreferrer">
                      <LinkedIn style={{ fontSize: '60px', marginLeft: "1rem", color: '#555'}} />
                    </a>
                  </span>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: "40rem", backgroundColor: "#d3d4d5" }}>
                <div className="person-content" style={{ paddingTop: "1rem"}}>
                <img src={Korhan} alt="korhan" style={{width: "100px", borderRadius: "50%"}}/>
                  <br/>
                  <b>Korhan Kemal Kaya</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                  </div>
                  <span>
                    <a href="https://github.com/korhankemalkaya" target="_blank" rel="noopener noreferrer">
                      <GitHub style={{ fontSize: '60px', color: '#555' }} />
                    </a>
                    <a href="https://www.linkedin.com/in/korhankemalkaya/" target="_blank" rel="noopener noreferrer">
                      <LinkedIn style={{ fontSize: '60px', marginLeft: "1rem", color: '#555'}} />
                    </a>
                  </span>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: "40rem", backgroundColor: "#d3d4d5" }}>
                <div className="person-content" style={{ paddingTop: "1rem"}}>
                <img src={Safa} alt="kaan" style={{width: "100px", borderRadius: "50%"}}/>
                  <br/>
                  <b>Safa Eren Kuday</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                    <br/>
                    <em>Bilkent MATH - Double Major</em>
                  </div>
                  <span>
                    <a href="https://github.com/safa54" target="_blank" rel="noopener noreferrer">
                      <GitHub style={{ fontSize: '60px', color: '#555' }} />
                    </a>
                    <a href="https://www.linkedin.com/in/safa-eren-kuday/" target="_blank" rel="noopener noreferrer">
                      <LinkedIn style={{ fontSize: '60px', marginLeft: "1rem", color: '#555'}} />
                    </a>
                  </span>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} >
              <Paper elevation={3} className="person-box" style={{ maxWidth: "40rem", backgroundColor: "#d3d4d5" }}>
                <div className="person-content" style={{ paddingTop: "1rem"}}>
                <img src={Kaan} alt="kaan" style={{width: "100px", borderRadius: "50%"}}/>
                  <br/>
                  <b>Kaan Tek</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                  </div>
                  <span>
                    <a href="https://github.com/KaanTekTr" target="_blank" rel="noopener noreferrer">
                      <GitHub style={{ fontSize: '60px', color: '#555' }} />
                    </a>
                    <a href="https://www.linkedin.com/in/kaan-tek-a299bb195/" target="_blank" rel="noopener noreferrer">
                      <LinkedIn style={{ fontSize: '60px', marginLeft: "1rem", color: '#555'}} />
                    </a>
                  </span>
                </div>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Paper elevation={3} className="person-box" style={{ maxWidth: "40rem", backgroundColor: "#d3d4d5" }}>
                <div className="person-content" style={{ paddingTop: "1rem"}}>
                <img src={Erkin} alt="erkin" style={{width: "100px", borderRadius: "50%"}}/>
                  <br/>
                  <b>Erkin Aydın</b>
                  <div>
                    <em>Bilkent CS - Senior Student</em>
                  </div>
                  <span>
                    <a href="https://github.com/Erkin-Aydin" target="_blank" rel="noopener noreferrer">
                      <GitHub style={{ fontSize: '60px', color: '#555' }} />
                    </a>
                    <a href="https://www.linkedin.com/in/erkin-ayd%C4%B1n-167aa8229/" target="_blank" rel="noopener noreferrer">
                      <LinkedIn style={{ fontSize: '60px', marginLeft: "1rem", color: '#555'}} />
                    </a>
                  </span>
                </div>
              </Paper>
            </Grid>
          </Grid>
        </Container>
        <h2>Our Supervisor</h2>
        <Container className="centered-container">
        <Grid item xs={12} sm={12} style={{display: "flex", alignItems: "center", justifyContent: "center"}}>
          <Paper elevation={3} className="person-box" style={{ maxWidth: 800, backgroundColor: "#d3d4d5", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center"  }}>
            <div className="person-content" style={{ paddingTop: "1rem"}}>
              <img src={ECicek} alt="e-cicek" style={{width: "125px", borderRadius: "50%"}}/>
              <br/>
              <b>Abdullah Ercüment Çiçek</b>
              <div>
                <em>Associate Professor at Bilkent CS</em>
                <br />
                <em>Adj. Faculty Member at the Computational Biology Department of School of Computer Science Carnegie Mellon University</em>
              </div>
              <div>
                <span>
                  <a href="http://ciceklab.cs.bilkent.edu.tr/" target="_blank" rel="noopener noreferrer">
                    <img src={CiceklabImage} alt="Link to Çiçeklab" />
                  </a>
                  <a href="https://github.com/Erkin-Aydin" target="_blank" rel="noopener noreferrer">
                    <GitHub style={{ fontSize: '60px', marginLeft: "2.5rem", color: '#555' }} />
                  </a>
                  <a href="https://www.linkedin.com/in/ercumentcicek/" target="_blank" rel="noopener noreferrer">
                    <LinkedIn style={{ fontSize: '60px', marginLeft: "1rem", color: '#555' }} />
                  </a>
                </span>
              </div>
            </div>
          </Paper>
        </Grid>
        </Container>
    </div>
  );
};

export default Team;
