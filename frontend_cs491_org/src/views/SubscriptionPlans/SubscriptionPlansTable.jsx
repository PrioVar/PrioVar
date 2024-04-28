import React, { useState, useEffect } from 'react';
import { Box, Card, CardContent, Typography, Grid, Button } from '@material-ui/core';
import axios from 'axios';
import { MIconButton } from '../../components/@material-extend';
import { useSnackbar } from 'notistack5';
import { Icon } from '@iconify/react';
import closeFill from '@iconify/icons-eva/close-fill'
import { ROOTS_PrioVar } from '../../routes/paths'


const SubscriptionPlansTable = () => {
  const { enqueueSnackbar, closeSnackbar } = useSnackbar()
  const healthCenterId = localStorage.getItem('healthCenterId') || '';
  const [remainingAnalyses, setRemainingAnalyses] = useState(null); 

  useEffect(() => {
    const fetchRemainingAnalyses = async () => {
      try {
        const response = await axios.get(`${ROOTS_PrioVar}/medicalCenter/${healthCenterId}`);
        setRemainingAnalyses(response.data.remainingAnalyses); 
      } catch (error) {
        console.error('Error fetching remaining analyses:', error);
        // handle error, possibly setting remainingAnalyses to an error state or default value
      }
    };
    fetchRemainingAnalyses();

  }, [healthCenterId]); 


  const plans = [
    {
      name: 'Junior Packet',
      price: '99.99$',
      features: ['Variant Analysis', 'Basic Report', '10 Patient Cases', 'Email Support'],
    },
    {
      name: 'Bioinformatician',
      price: '149.99$',
      features: ['Variant Analysis', 'Comprehensive Report', '20 Patient Cases', 'Phone Support'],
    },
    {
      name: 'Devourer Of The Genes',
      price: '199.99$',
      features: ['Variant Analysis', 'Comprehensive Report', '30 Patient Cases', '24/7 VIP Support'],
    },
  ];

  const HandleSelectPlan = async (subscriptionId) => {
    const url = `${ROOTS_PrioVar}/addSubscription/${healthCenterId}/${subscriptionId}`;
    try {
      const response = await axios.post(url);
      console.log("SUCCESS")
      //console.log(response.data);
      // handle response
      enqueueSnackbar('Subscription added successfully!', {
        variant: 'success',
        action: (key) => (
            <MIconButton size="small" onClick={() => {
              closeSnackbar(key);
              // Reload the page after the request is successful
              window.location.reload();
            }}>
              <Icon icon={closeFill} />
            </MIconButton>
          ),
      })
    } catch (error) {
      console.log("ERROR")
      console.error(error);
      // handle error
    }
  };

  return (
    <Box p={4} style={{
      position: 'absolute', 
      top: 0, 
      left: 0, 
      width: '100%', 
      height: '100vh', 
      backgroundImage: 'url("/static/new_images/dna-helix-removebg.png")', 
      backgroundSize: 'cover', 
      backgroundPosition: 'center center',
      backgroundColor: 'rgba(255, 255, 255, 0.5)', // Adds white transparency
      backgroundBlendMode: 'overlay' // This blends the background color with the image
    }}>
      <Typography variant="h4" gutterBottom align="center" mt={8}>
        Subscription Plans
      </Typography>
      <Grid container spacing={2} justifyContent="center" mt={4}>
        {plans.map((plan, index) => (
          <Grid item key={index} xs={12} sm={4}>
            <Card raised>
              <CardContent>
                <Typography variant="h5" component="h2" align="center" gutterBottom>
                  {plan.name}
                </Typography>
                <Box>
                  {plan.features.map((feature, idx) => (
                    <Typography key={idx} variant="body2">
                      <li key={idx}>{feature}</li>
                    </Typography>
                  ))}
                </Box>
                <Typography mt={4} variant="h6" align="center" gutterBottom >
                  {plan.price} / month
                </Typography>
                <Button
                  variant="contained"
                  color="primary"
                  fullWidth
                  onClick={() => HandleSelectPlan(index+1)}
                >
                  Select
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
      <Box mt={2} ml={2} align="left">
        <Typography variant="subtitle1">
          Number of remaining analyses for your plan: {remainingAnalyses !== null ? remainingAnalyses : 'Loading...'}
        </Typography>
      </Box>
    </Box>
  );
};

export default SubscriptionPlansTable;
