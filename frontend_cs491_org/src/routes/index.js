import { lazy, Suspense } from 'react'
import { Navigate, useLocation, useRoutes } from 'react-router-dom'
import LogoOnlyLayout from 'src/layouts/LogoOnlyLayout'

// import RoleBasedGuard from '../guards/RoleBasedGuard';
import SideNav from 'src/components/sidenav'
// components
import LoadingScreen from '../components/LoadingScreen'
import AuthGuard from '../guards/AuthGuard'
// guards
import GuestGuard from '../guards/GuestGuard'
import DashboardLayout from '../layouts/dashboard'
// layouts
import MainLayout from '../layouts/main'
import FileUploadView from '../views/FileUploadView'

import ClinicsPatients from '../views/ClinicsPatients'
import MyPatients from '../views/MyPatients'
import CustomQuery from 'src/views/CustomQuery'
import SubscriptionPlans from 'src/views/SubscriptionPlans'
import SimilarPatients from 'src/views/SimilarPatients'
import AddClinician from 'src/views/AddClinicianView'
// ----------------------------------------------------------------------

const Loadable = (Component) =>
  function (props) {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const { pathname } = useLocation()
    const isDashboard = pathname.includes('/dashboard')

    return (
      <Suspense
        fallback={
          <LoadingScreen
            sx={{
              ...(!isDashboard && {
                top: 0,
                left: 0,
                width: 1,
                zIndex: 9999,
                position: 'fixed',
              }),
            }}
          />
        }
      >
        <Component {...props} />
      </Suspense>
    )
  }

export default function Router() {
  return useRoutes([
    {
      path: '/',
      children: [
        {
          path: '',
          element: (
            <GuestGuard>
              <Login />
            </GuestGuard>
          ),
        },
        {
          path: 'login-health-center',
          element: (
            <GuestGuard>
              <LoginHealthCenter />
            </GuestGuard>
          ),
        },
        {
          path: 'login-admin',
          element: (
            <GuestGuard>
              <LoginAdmin />
            </GuestGuard>
          ),
        },
        {
          path: 'register',
          element: (
            <GuestGuard>
              <Register />
            </GuestGuard>
          ),
        },
        { path: 'login-unprotected', element: <Login /> },
        { path: 'login-health-center-unprotected', element: <LoginHealthCenter /> },
        { path: 'login-admin-unprotected', element: <LoginAdmin /> },
        { path: 'register-unprotected', element: <Register /> },
        { path: 'reset-password', element: <ResetPassword /> },
        { path: 'verify', element: <VerifyCode /> },
      ],
    },
    {
      path: 'libra',
      element: (
        <AuthGuard>
          <SideNav />
          <DashboardLayout />
        </AuthGuard>
      ),
      children: [
        { path: '/', element: <Navigate to="/libra/files" replace /> },
        { path: 'files', element: <FileUploadView /> },
        // TODO: Make sampleName optional
        { path: '/variants/:fileId/:sampleName', element: <VariantsView /> },
        { path: '/variants/:fileId/:sampleName/:chrom/:pos', element: <VariantDetailsView /> },
        { path: '/sample/:fileId/:sampleName', element: <VariantDashboard /> },
        { path: '/clinics/:healthCenterId/patients', element: <ClinicsPatients />},
        { path: '/clinician/:healthCenterId/patients', element: <MyPatients />},
        { path: '/customquery', element: < CustomQuery/>},
        { path: '/subscriptionPlans', element: < SubscriptionPlans/>},
        { path: '/similarPatients', element: < SimilarPatients/>},
        { path: '/addClinician', element: < AddClinician/>},
        {
          path: 'user',
          children: [
            { path: '/', element: <Navigate to="/libra/user/list" replace /> },
            //      { path: 'profile', element: <UserProfile /> },
            //    { path: 'cards', element: <UserCards /> },
            { path: 'list', element: <UserList /> },
            { path: 'new', element: <UserCreate /> },
            { path: '/:id/edit', element: <UserCreate /> },
            { path: 'account', element: <UserAccount /> },
          ],
        },
      ],
    },
    
    // Main Routes
    {
      path: '*',
      element: <LogoOnlyLayout />,
      children: [
        { path: 'coming-soon', element: <ComingSoon /> },
        { path: 'maintenance', element: <Maintenance /> },
        { path: '500', element: <Page500 /> },
        { path: '404', element: <NotFound /> },
        { path: '*', element: <Navigate to="/404" replace /> },
      ],
    },
    /*
    {
      path: '/',
      element: <MainLayout />,
      
      children: [
        { path: '/', element: <Login /> },
        { path: 'about-us', element: <About /> },
        { path: 'contact-us', element: <Contact /> },
        { path: 'faqs', element: <Faqs /> },
      ],
      
    },
    */
  ])
}

// IMPORT COMPONENTS

// Authentication
const Login = Loadable(lazy(() => import('../pages/authentication/Login')))
const LoginHealthCenter = Loadable(lazy(() => import('../pages/authentication/LoginHealthCenter')))
const LoginAdmin = Loadable(lazy(() => import('../pages/authentication/LoginAdmin')))
const Register = Loadable(lazy(() => import('../pages/authentication/Register')))
const ResetPassword = Loadable(lazy(() => import('../pages/authentication/ResetPassword')))
const VerifyCode = Loadable(lazy(() => import('../pages/authentication/VerifyCode')))
// Dashboard
const GeneralApp = Loadable(lazy(() => import('../pages/dashboard/GeneralApp')))
const GeneralEcommerce = Loadable(lazy(() => import('../pages/dashboard/GeneralEcommerce')))
const GeneralAnalytics = Loadable(lazy(() => import('../pages/dashboard/GeneralAnalytics')))
const GeneralBanking = Loadable(lazy(() => import('../pages/dashboard/GeneralBanking')))
const GeneralBooking = Loadable(lazy(() => import('../pages/dashboard/GeneralBooking')))
const EcommerceShop = Loadable(lazy(() => import('../pages/dashboard/EcommerceShop')))
const EcommerceProductDetails = Loadable(lazy(() => import('../pages/dashboard/EcommerceProductDetails')))
const EcommerceProductList = Loadable(lazy(() => import('../pages/dashboard/EcommerceProductList')))
const EcommerceProductCreate = Loadable(lazy(() => import('../pages/dashboard/EcommerceProductCreate')))
const EcommerceCheckout = Loadable(lazy(() => import('../pages/dashboard/EcommerceCheckout')))
const EcommerceInvoice = Loadable(lazy(() => import('../pages/dashboard/EcommerceInvoice')))
const BlogPosts = Loadable(lazy(() => import('../pages/dashboard/BlogPosts')))
const BlogPost = Loadable(lazy(() => import('../pages/dashboard/BlogPost')))
const BlogNewPost = Loadable(lazy(() => import('../pages/dashboard/BlogNewPost')))
const UserProfile = Loadable(lazy(() => import('../pages/dashboard/UserProfile')))
const UserCards = Loadable(lazy(() => import('../pages/dashboard/UserCards')))
const UserList = Loadable(lazy(() => import('../pages/dashboard/UserList')))
const UserAccount = Loadable(lazy(() => import('../pages/dashboard/UserAccount')))
const UserCreate = Loadable(lazy(() => import('../pages/dashboard/UserCreate')))
const Chat = Loadable(lazy(() => import('../pages/dashboard/Chat')))
const Mail = Loadable(lazy(() => import('../pages/dashboard/Mail')))
const Calendar = Loadable(lazy(() => import('../pages/dashboard/Calendar')))
const Kanban = Loadable(lazy(() => import('../pages/dashboard/Kanban')))
// Main
const LandingPage = Loadable(lazy(() => import('../pages/LandingPage')))
const About = Loadable(lazy(() => import('../pages/About')))
const Contact = Loadable(lazy(() => import('../pages/Contact')))
const Faqs = Loadable(lazy(() => import('../pages/Faqs')))
const ComingSoon = Loadable(lazy(() => import('../pages/ComingSoon')))
const Maintenance = Loadable(lazy(() => import('../pages/Maintenance')))
const Pricing = Loadable(lazy(() => import('../pages/Pricing')))
const Payment = Loadable(lazy(() => import('../pages/Payment')))
const Page500 = Loadable(lazy(() => import('../pages/Page500')))
const NotFound = Loadable(lazy(() => import('../pages/Page404')))
// Components
const ComponentsOverview = Loadable(lazy(() => import('../pages/ComponentsOverview')))
const Color = Loadable(lazy(() => import('../pages/components-overview/foundations/FoundationColors')))
const Typography = Loadable(lazy(() => import('../pages/components-overview/foundations/FoundationTypography')))
const Shadows = Loadable(lazy(() => import('../pages/components-overview/foundations/FoundationShadows')))
const Grid = Loadable(lazy(() => import('../pages/components-overview/foundations/FoundationGrid')))
const Icons = Loadable(lazy(() => import('../pages/components-overview/foundations/FoundationIcons')))
const Accordion = Loadable(lazy(() => import('../pages/components-overview/material-ui/Accordion')))
const Alert = Loadable(lazy(() => import('../pages/components-overview/material-ui/Alert')))
const Autocomplete = Loadable(lazy(() => import('../pages/components-overview/material-ui/Autocomplete')))
const Avatar = Loadable(lazy(() => import('../pages/components-overview/material-ui/Avatar')))
const Badge = Loadable(lazy(() => import('../pages/components-overview/material-ui/Badge')))
const Breadcrumb = Loadable(lazy(() => import('../pages/components-overview/material-ui/Breadcrumb')))
const Buttons = Loadable(lazy(() => import('../pages/components-overview/material-ui/buttons')))
const Checkbox = Loadable(lazy(() => import('../pages/components-overview/material-ui/Checkboxes')))
const Chip = Loadable(lazy(() => import('../pages/components-overview/material-ui/chips')))
const Dialog = Loadable(lazy(() => import('../pages/components-overview/material-ui/dialog')))
const Label = Loadable(lazy(() => import('../pages/components-overview/material-ui/Label')))
const List = Loadable(lazy(() => import('../pages/components-overview/material-ui/Lists')))
const Menu = Loadable(lazy(() => import('../pages/components-overview/material-ui/Menus')))
const Pagination = Loadable(lazy(() => import('../pages/components-overview/material-ui/Pagination')))
const Pickers = Loadable(lazy(() => import('../pages/components-overview/material-ui/pickers')))
const Popover = Loadable(lazy(() => import('../pages/components-overview/material-ui/Popover')))
const Progress = Loadable(lazy(() => import('../pages/components-overview/material-ui/progress')))
const RadioButtons = Loadable(lazy(() => import('../pages/components-overview/material-ui/RadioButtons')))
const Rating = Loadable(lazy(() => import('../pages/components-overview/material-ui/Rating')))
const Slider = Loadable(lazy(() => import('../pages/components-overview/material-ui/Slider')))
const Snackbar = Loadable(lazy(() => import('../pages/components-overview/material-ui/Snackbar')))
const Stepper = Loadable(lazy(() => import('../pages/components-overview/material-ui/stepper')))
const Switches = Loadable(lazy(() => import('../pages/components-overview/material-ui/Switches')))
const Table = Loadable(lazy(() => import('../pages/components-overview/material-ui/table')))
const Tabs = Loadable(lazy(() => import('../pages/components-overview/material-ui/Tabs')))
const Textfield = Loadable(lazy(() => import('../pages/components-overview/material-ui/textfield')))
const Timeline = Loadable(lazy(() => import('../pages/components-overview/material-ui/Timeline')))
const Tooltip = Loadable(lazy(() => import('../pages/components-overview/material-ui/Tooltip')))
const TransferList = Loadable(lazy(() => import('../pages/components-overview/material-ui/transfer-list')))
const TreeView = Loadable(lazy(() => import('../pages/components-overview/material-ui/TreeView')))
const DataGrid = Loadable(lazy(() => import('../pages/components-overview/material-ui/data-grid')))
//
const Charts = Loadable(lazy(() => import('../pages/components-overview/extra/Charts')))
const Map = Loadable(lazy(() => import('../pages/components-overview/extra/Map')))
const Editor = Loadable(lazy(() => import('../pages/components-overview/extra/Editor')))
const CopyToClipboard = Loadable(lazy(() => import('../pages/components-overview/extra/CopyToClipboard')))
const Upload = Loadable(lazy(() => import('../pages/components-overview/extra/Upload')))
const Carousel = Loadable(lazy(() => import('../pages/components-overview/extra/Carousel')))
const MultiLanguage = Loadable(lazy(() => import('../pages/components-overview/extra/MultiLanguage')))
const Animate = Loadable(lazy(() => import('../pages/components-overview/extra/animate')))
const MegaMenu = Loadable(lazy(() => import('../pages/components-overview/extra/MegaMenu')))
const FormValidation = Loadable(lazy(() => import('../pages/components-overview/extra/form-validation')))

const VariantsView = Loadable(lazy(() => import('../views/VariantsView')))
const VariantDetailsView = Loadable(lazy(() => import('../views/VariantDetailsView')))
const VariantDashboard = Loadable(lazy(() => import('../views/VariantDashboard')))
