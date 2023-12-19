import linkedinFill from '@iconify/icons-eva/linkedin-fill'
import { Icon } from '@iconify/react'
import { Container, Divider, Grid, IconButton, Link, Stack, Typography } from '@material-ui/core'
// material
import { styled } from '@material-ui/core/styles'
import { Link as RouterLink } from 'react-router-dom'
import { Link as ScrollLink } from 'react-scroll'
//
import Logo from '../../components/Logo'
// routes
import { PATH_PAGE } from '../../routes/paths'

// ----------------------------------------------------------------------

const SOCIALS = [{ name: 'Linkedin', icon: linkedinFill, href: 'https://www.linkedin.com/company/87409820/' }]

const LINKS = [
  {
    headline: 'Company',
    children: [
      { name: 'About us', href: PATH_PAGE.about },
      { name: 'Contact us', href: PATH_PAGE.contact },
      // { name: 'FAQs', href: PATH_PAGE.faqs },
    ],
  },
  {
    headline: 'Legal',
    children: [
      { name: 'Terms and Condition', href: '/coming-soon' },
      { name: 'Privacy Policy', href: '/coming-soon' },
    ],
  },
  {
    headline: 'Contact',
    children: [
      { name: 'info@lidyagenomics.com', href: 'mailto:info@lidyagenomics.com', isMail: true },
      { name: 'PrioVar İthalat İhracat Ltd. Şti.', href: '#' },
      { name: 'Bilkent, Ankara, Turkey', href: '#' },
    ],
  },
]

const RootStyle = styled('div')(({ theme }) => ({
  position: 'relative',
  backgroundColor: theme.palette.background.default,
}))

// ----------------------------------------------------------------------

export default function MainFooter() {
  return (
    <RootStyle>
      <Divider />
      <Container maxWidth="lg" sx={{ pt: 10 }}>
        <Grid
          container
          justifyContent={{ xs: 'center', md: 'space-between' }}
          sx={{ textAlign: { xs: 'center', md: 'left' } }}
        >
          <Grid item xs={12} sx={{ mb: 3 }}>
            <ScrollLink to="move_top" spy smooth>
              <Logo sx={{ mx: { xs: 'auto', md: 'inherit' } }} />
            </ScrollLink>
          </Grid>
          <Grid item xs={8} md={3}>
            <Typography variant="body2" sx={{ pr: { md: 5 } }}>
              PrioVar aims to provide powerful solutions to the problems in disease genomics using computer
              science.
            </Typography>

            <Stack
              spacing={1.5}
              direction="row"
              justifyContent={{ xs: 'center', md: 'flex-start' }}
              sx={{ mt: 5, mb: { xs: 5, md: 0 } }}
            >
              {SOCIALS.map((social) => (
                <IconButton key={social.name} color="primary" sx={{ p: 1 }} href={social.href} target="_blank">
                  <Icon icon={social.icon} width={16} height={16} href={social.href} target="_blank" />
                  <Link href={social.href} target="_blank"></Link>
                </IconButton>
              ))}
            </Stack>
          </Grid>

          <Grid item xs={12} md={7}>
            <Stack spacing={5} direction={{ xs: 'column', md: 'row' }} justifyContent="space-between">
              {LINKS.map((list) => {
                const { headline, children } = list
                return (
                  <Stack key={headline} spacing={2}>
                    <Typography component="p" variant="overline">
                      {headline}
                    </Typography>
                    {children.map((link) =>
                      link.isMail ? (
                        <Link
                          href={link.href}
                          key={link.name}
                          color="inherit"
                          variant="body2"
                          sx={{ display: 'block' }}
                        >
                          {link.name}
                        </Link>
                      ) : (
                        <Link
                          to={link.href}
                          key={link.name}
                          color="inherit"
                          variant="body2"
                          component={RouterLink}
                          sx={{ display: 'block' }}
                        >
                          {link.name}
                        </Link>
                      ),
                    )}
                  </Stack>
                )
              })}
            </Stack>
          </Grid>
        </Grid>

        <Typography
          component="p"
          variant="body2"
          sx={{
            mt: 10,
            pb: 5,
            fontSize: 13,
            textAlign: { xs: 'center', md: 'left' },
          }}
        >
          PrioVar © 2021. All rights reserved.
        </Typography>
      </Container>
    </RootStyle>
  )
}
