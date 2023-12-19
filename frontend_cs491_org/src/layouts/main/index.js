// material
import { Outlet, useLocation } from 'react-router-dom'
// components
import MainFooter from './MainFooter'
//
import MainNavbar from './MainNavbar'
import Login from '../../pages/authentication/Login'
// ----------------------------------------------------------------------

export default function MainLayout() {
  const { pathname } = useLocation()
  const isHome = pathname === '/'

  /*
  return (
    <>
      <MainNavbar />
      <div>
        <Outlet />
      </div>

    </>
  )
  */
  return (
    <>
      <Login />
    </>
  )
}
