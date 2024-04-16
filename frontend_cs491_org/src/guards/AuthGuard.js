import PropTypes from 'prop-types'
import { useState } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
// hooks
import useAuth from '../hooks/useAuth'
// pages
import Login from '../pages/authentication/Login'

// ----------------------------------------------------------------------

AuthGuard.propTypes = {
  children: PropTypes.node,
}

export default function AuthGuard({ children }) {
  const { isAuthenticated } = useAuth()
  const { pathname } = useLocation()
  const [requestedLocation, setRequestedLocation] = useState(null)

  if (!isAuthenticated) {
    if (pathname !== requestedLocation) {
      setRequestedLocation(pathname)
    }
    return <Login />
  }

  if (requestedLocation && pathname !== requestedLocation) {
    // TODO: Remove
    if (requestedLocation === '/dashboard') {
      window.location.href = 'http://localhost:3000/app/samples'
      return null
      //return <Navigate to={'http://localhost:3000/app/samples'} />
    }
    setRequestedLocation(null)
    return <Navigate to={requestedLocation} />
  }

  return <>{children}</>
}
