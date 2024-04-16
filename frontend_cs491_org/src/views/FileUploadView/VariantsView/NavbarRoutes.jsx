import { Button, Divider } from '@material-ui/core'
import { useTheme } from '@material-ui/styles'

const NavbarRoutes = function ({ navConfig, params }) {
  const theme = useTheme()
  return navConfig.map((nav) => {
    if (nav.title === 'Variants') {
      nav.path = nav.path.replace(':fileId', params.fileId).replace(':sampleName', params.sampleName)
    }
    if (nav.title === 'Variant Dashboard') {
      nav.path = nav.path
        .replace(':fileId', localStorage.getItem('dashboardSampleID'))
        .replace(':sampleName', params.sampleName)
    }
    const { title, path, icon } = nav
    return (
      <>
        <Button
          variant="primary"
          sx={{ background: theme.palette.background.default, color: theme.palette.text.primary }}
          startIcon={icon}
          onClick={() => (window.location.href = path)}
        >
          {title}
        </Button>
        <Divider orientation="vertical" flexItem />
      </>
    )
  })
}

export default NavbarRoutes
