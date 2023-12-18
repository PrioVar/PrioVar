import React from 'react';
import { Box, Card, CardContent, Typography, Grid, Button } from '@material-ui/core';
import axios from 'axios';


const SubscriptionPlansTable = () => {

  const healthCenterId = localStorage.getItem('healthCenterId') || ''; 


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

  const handleSelectPlan = async (subscriptionId) => {
    const url = `http://localhost:8080/medicalCenter/addSubscription/${healthCenterId}/${subscriptionId}`;
    try {
      const response = await axios.post(url);
      console.log("SUCCESS")
      console.log(response.data);
      // handle response
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
                  onClick={() => handleSelectPlan(index+1)}
                >
                  Select
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default SubscriptionPlansTable;
