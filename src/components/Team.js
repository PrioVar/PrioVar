import React, { useEffect } from 'react';
import { Grid, Paper, Container } from '@mui/material';
import { GitHub } from '@mui/icons-material';
import './Team.css'; // Import the CSS file for styling

const Team = () => {
  // Sample project data
  useEffect(() => {
    document.title = 'Meet the Xaga Team'
  }, []);
  const team = [
    {
      name: 'Alperen Gözeten',
      explanation: 'Small explanation goes here.',
      student: true,
      link: 'https://github.com/alperengozeten',
    },
    {
      name: 'Korhan Kemal Kaya',
      explanation: 'Small explanation goes here.',
      student: true,
      link: 'https://github.com/korhankemalkaya'
    },
    {
      name: 'Safa Eren Kuday',
      explanation: 'Small explanation goes here.',
      student: true,
      link: 'https://github.com/safa54'
    },
    {
      name: 'Kaan Tek',
      explanation: 'Small explanation goes here.',
      student: true,
      link: 'https://github.com/KaanTekTr'
    },
    {
      name: 'Erkin Aydın',
      explanation: 'Small explanation goes here.',
      student: true,
      link: 'https://github.com/Erkin-Aydin'
    },
    {
      name: 'Abdullah Ercüment Çiçek',
      explanation: 'Associate Professor at the Computer Engineering Department of Bilkent University',
      student: false,
      link: 'http://ciceklab.cs.bilkent.edu.tr/'
    },
    {
      name: 'Innovation Expert name goes here',
      explanation: 'Small explanation goes here.',
      student: null,
    },
  ];

  // Group projects based on their status
  const teamMembers = {
    'Xaga': team.filter((person) => person.student === true),
    'Supervisor': team.filter((person) => person.student === false),
    'Innovation Expert': team.filter((person) => person.student === null),
  };

  return (
    <div>
      <h2>Meet the Xaga Team</h2>
      {Object.entries(teamMembers).map(([group, people]) => (
        <div key={group}>
          <h3>{group}</h3>
          <Container className="centered-container">
            <Grid container spacing={2}>
              {people.map((person, index) => (
                <Grid item xs={12} sm={6} key={index}>
                  <Paper elevation={3} className="person-box" style={{ maxWidth: 800 }}>
                    <div className="person-content">
                      <p>
                        <b>{person.name}</b>
                        <p>{person.explanation}</p>
                      </p>
                      {person.student ? ( // Check if the person is a student
                        <a href={person.link} target="_blank" rel="noopener noreferrer">
                          <GitHub /> {/* Display GitHub icon for students */}
                        </a>
                      ) : (
                        <a href={person.link} target="_blank" rel="noopener noreferrer">
                          <img src="./ciceklab.png" alt="Link to Person" /> {/* Display image for non-students */}
                        </a>
                      )}
                    </div>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Container>
        </div>
      ))}
    </div>
  );
};

export default Team;
