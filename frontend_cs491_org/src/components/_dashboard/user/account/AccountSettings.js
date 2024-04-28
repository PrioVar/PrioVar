import { /*Card,*/ Paper, Stack, Typography } from '@material-ui/core'
import { useState } from 'react'
import Scrollbar from 'src/components/Scrollbar'
import SettingFullscreen from 'src/components/settings/SettingFullscreen'
import SettingMode from 'src/components/settings/SettingMode'

const DRAWER_WIDTH = 500

export default function AccountSettings() {
  const [open] = useState(true)

  return (
    <Stack
      sx={{
        direction: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Paper
        sx={{
          height: 1,
          width: '0px',
          overflow: 'hidden',
          boxShadow: (theme) => theme.customShadows.z24,
          transition: (theme) => theme.transitions.create('width'),
          ...(open && { width: DRAWER_WIDTH }),
        }}
      >
        <Scrollbar sx={{ height: 1 }}>
          <Stack spacing={4} sx={{ pt: 3, px: 3, pb: 15 }}>
            <Stack spacing={1.5}>
              <Typography variant="subtitle2">Mode</Typography>
              <SettingMode />
            </Stack>

            <SettingFullscreen />
          </Stack>
        </Scrollbar>
      </Paper>
    </Stack>
  )
}
