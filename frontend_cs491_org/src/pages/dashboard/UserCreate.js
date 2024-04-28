import { useEffect } from 'react'
//import { paramCase } from 'change-case'
import { useParams, useLocation } from 'react-router-dom'
// material
import { Container } from '@material-ui/core'
// redux
import { useDispatch, useSelector } from '../../redux/store'
import { getUserList } from '../../redux/slices/user'
// routes
//import { PATH_DASHBOARD } from '../../routes/paths'
// hooks
import useSettings from '../../hooks/useSettings'
// components
import Page from '../../components/Page'
import HeaderBreadcrumbs from '../../components/HeaderBreadcrumbs'
import UserNewForm from '../../components/_dashboard/user/UserNewForm'

// ----------------------------------------------------------------------

export default function UserCreate() {
  const { themeStretch } = useSettings()
  const dispatch = useDispatch()
  const { pathname } = useLocation()
  const { id } = useParams()
  const { userList } = useSelector((state) => state.user)
  const isEdit = typeof id === 'string'

  useEffect(() => {
    dispatch(getUserList())
  }, [dispatch])

  return (
    <Page title="User: Create a new user | PrioVar">
      <Container maxWidth={themeStretch ? false : 'lg'}>
        <HeaderBreadcrumbs
          heading={!isEdit ? 'Create a new user' : 'Edit user'}
          // links={[
          //   { name: 'Dashboard', href: PATH_DASHBOARD.root },
          //   { name: 'User', href: PATH_DASHBOARD.user.root },
          //   { name: !isEdit ? 'New user' : id },
          // ]}
        />

        <UserNewForm isEdit={isEdit} currentUser={{}} />
      </Container>
    </Page>
  )
}
