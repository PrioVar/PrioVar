// material
import { Divider, Link, Stack, Typography } from '@material-ui/core'
//
import { MotionInView, varFadeInUp } from '../../animate'

// ----------------------------------------------------------------------

export default function ContactEmail() {
  return (
    <Stack spacing={1}>
      <MotionInView variants={varFadeInUp}>
        <Typography variant="h4">Email us</Typography>
      </MotionInView>

      <Divider sx={{ borderStyle: 'dashed' }} />

      <MotionInView variants={varFadeInUp}>
        <Link href="mailto:info@lidyagenomics.com" variant="h5">
          info@lidyagenomics.com
        </Link>
      </MotionInView>
    </Stack>
  )
}
