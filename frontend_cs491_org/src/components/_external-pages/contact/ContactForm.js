// material
import { Button, Stack, TextField, Typography } from '@material-ui/core'
//
import { MotionInView, varFadeInUp } from '../../animate'

// ----------------------------------------------------------------------

export default function ContactForm() {
  return (
    <Stack spacing={5}>
      <MotionInView variants={varFadeInUp}>
        <Typography variant="h3">We'll be glad to set up a meeting and schedule a demo!</Typography>
      </MotionInView>

      <Stack spacing={3}>
        <MotionInView variants={varFadeInUp}>
          <TextField fullWidth label="Name" />
        </MotionInView>

        <MotionInView variants={varFadeInUp}>
          <TextField fullWidth label="Email" />
        </MotionInView>

        <MotionInView variants={varFadeInUp}>
          <TextField fullWidth label="Subject" />
        </MotionInView>

        <MotionInView variants={varFadeInUp}>
          <TextField fullWidth label="Enter your message here." multiline rows={4} />
        </MotionInView>
      </Stack>

      <MotionInView variants={varFadeInUp}>
        <Button size="large" variant="contained">
          Submit Now
        </Button>
      </MotionInView>
    </Stack>
  )
}
