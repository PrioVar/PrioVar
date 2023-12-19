// material
import { Outlet, useLocation } from 'react-router-dom'
// components
import MainFooter from './MainFooter'
//
import MainNavbar from './MainNavbar'
import { G } from '@react-pdf/renderer'
import { BrowserRouter as Navigate } from 'react-router-dom';
// ----------------------------------------------------------------------

export default function MainLayout() {
  const { pathname } = useLocation()
  // Check if the current path is the root path
    return <Navigate to="/auth/login" />;
}
