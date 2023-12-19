import React, { useState, useEffect } from 'react';
import { Box, Card, CardContent, Typography, Grid, Button } from '@material-ui/core';
import axios from 'axios';
import { MIconButton } from '../../components/@material-extend';
import { useSnackbar } from 'notistack5';
import { Icon } from '@iconify/react';
import closeFill from '@iconify/icons-eva/close-fill'


const SubscriptionPlansTable = () => {
  const { enqueueSnackbar, closeSnackbar } = useSnackbar()
  const healthCenterId = localStorage.getItem('healthCenterId') || '';
  const [remainingAnalyses, setRemainingAnalyses] = useState(null); 

  useEffect(() => {
    const fetchRemainingAnalyses = async () => {
      try {
        const response = await axios.get(`http://localhost:8080/medicalCenter/${healthCenterId}`);
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
      price: '109.99$',
      features: ['Property 1', 'Property 2', 'Property 3'],
    },
    {
      name: 'Bioinformatician',
      price: '509.99$',
      features: ['Property 1', 'Property 2', 'Property 3', 'Property 4'],
    },
    {
      name: 'DEVOURER OF THE GENES',
      price: '1009.99$',
      features: ['Property 1', 'Property 2', 'Property 3', 'Property 4', 'Property 5'],
    },
  ];

  const HandleSelectPlan = async (subscriptionId) => {
    const url = `http://localhost:8080/medicalCenter/addSubscription/${healthCenterId}/${subscriptionId}`;
    try {
      const response = await axios.post(url);
      console.log("SUCCESS")
      console.log(response.data);
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
    <Box p={4} bgcolor="background.default">
      <Typography variant="h4" gutterBottom align="center" mt={4}>
        Subscription Plans
      </Typography>
      <Grid container spacing={4} justifyContent="center" mt={4}>
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
                      {feature}
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
