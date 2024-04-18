import { Typography, useMediaQuery } from '@material-ui/core';
import { styled, useTheme } from '@material-ui/core/styles';
import PropTypes from 'prop-types';
import { Link as RouterLink } from 'react-router-dom';
import { MHidden } from '../components/@material-extend';
import Logo from '../components/Logo';
/*
const HeaderStyle = styled('header')(({ theme }) => ({
  top: 0,
  zIndex: 1000,
  lineHeight: 0,
  width: '100%',
  display: 'flex',
  alignItems: 'center',
  position: 'fixed',
  padding: theme.spacing(3),
  justifyContent: 'space-between',
  [theme.breakpoints.up('md')]: {
    alignItems: 'flex-start',
    padding: theme.spacing(3, 5, 0, 3),
  },
}));
*/
AuthLayout.propTypes = {
  children: PropTypes.node,
};

export default function AuthLayout({ children }) {
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('lg'));
  const LogoContainer = styled('div')(({ theme }) => ({
    position: 'fixed',
    left: 0,
    top: '50%',
    transform: 'translateY(-50%)',
    padding: theme.spacing(3),
    left: theme.spacing(10),
  }));

  return (
   <>
   <RouterLink to="/">
        <LogoContainer>
            <RouterLink to="/">
            <Logo size={isSmallScreen ? 'small' : 'large'} />
            </RouterLink>
        </LogoContainer>
      </RouterLink>

      <MHidden width="smUp">
        <Typography
          variant="body2"
          sx={{
            mt: { md: -2 },
          }}
        >
          {children}
        </Typography>
      </MHidden>
   </>
  );
}
