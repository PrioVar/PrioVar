//

// routes
import { QueryClient, QueryClientProvider } from 'react-query'
import Settings from 'src/components/settings'
//import GoogleAnalytics from './components/GoogleAnalytics'
import LoadingScreen from './components/LoadingScreen'
import NotistackProvider from './components/NotistackProvider'
// components
// hooks
import useAuth from './hooks/useAuth'
import Router from './routes'
// theme
import ThemeConfig from './theme'
import { enableMapSet } from 'immer'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
    },
  },
})

enableMapSet()

// ----------------------------------------------------------------------

export default function App() {
  const { isInitialized } = useAuth()

  return (
    <ThemeConfig>
      <NotistackProvider>
        <Settings />
        {/* <GoogleAnalytics /> */}
        <QueryClientProvider client={queryClient}>{isInitialized ? <Router /> : <LoadingScreen />}</QueryClientProvider>
      </NotistackProvider>
    </ThemeConfig>
  )
}
