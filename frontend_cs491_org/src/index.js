// mock api
// material
import AdapterDateFns from '@material-ui/lab/AdapterDateFns'
import LocalizationProvider from '@material-ui/lab/LocalizationProvider'

// lazy image
import 'lazysizes'
import 'lazysizes/plugins/attrchange/ls.attrchange'
import 'lazysizes/plugins/object-fit/ls.object-fit'
import 'lazysizes/plugins/parent-fit/ls.parent-fit'

// map
import 'mapbox-gl/dist/mapbox-gl.css'

import ReactDOM from 'react-dom'
import 'react-draft-wysiwyg/dist/react-draft-wysiwyg.css'
import { HelmetProvider } from 'react-helmet-async'

// lightbox
import 'react-image-lightbox/style.css'

// editor
import 'react-quill/dist/quill.snow.css'
import { Provider as ReduxProvider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
import { PersistGate } from 'redux-persist/lib/integration/react'

// scroll bar
import 'simplebar/src/simplebar.css'
import 'slick-carousel/slick/slick-theme.css'

// slick-carousel
import 'slick-carousel/slick/slick.css'
// import { AuthProvider } from './contexts/JWTContext'
// import { AuthProvider } from './contexts/FirebaseContext';
// import { AuthProvider } from './contexts/AwsCognitoContext';
// import { AuthProvider } from './contexts/Auth0Context';
//
import App from './App'
// components
import LoadingScreen from './components/LoadingScreen'
import { CollapseDrawerProvider } from './contexts/CollapseDrawerContext'
// contexts
import { SettingsProvider } from './contexts/SettingsContext'

import { AuthProvider } from './contexts/TokenContext'

// i18n
import './locales/i18n'
// redux
import { persistor, store } from './redux/store'
import reportWebVitals from './reportWebVitals'
import * as serviceWorkerRegistration from './serviceWorkerRegistration'

// highlight
import './utils/highlight'

// ----------------------------------------------------------------------


ReactDOM.render(
  <HelmetProvider>
    <ReduxProvider store={store}>
      <PersistGate loading={<LoadingScreen />} persistor={persistor}>
        <LocalizationProvider dateAdapter={AdapterDateFns}>
          <SettingsProvider>
            <CollapseDrawerProvider>
              <BrowserRouter>
                <AuthProvider>
                  <App />
                </AuthProvider>
              </BrowserRouter>
            </CollapseDrawerProvider>
          </SettingsProvider>
        </LocalizationProvider>
      </PersistGate>
    </ReduxProvider>
  </HelmetProvider>,
  document.getElementById('root'),
)

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://cra.link/PWA
serviceWorkerRegistration.unregister()

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals()
