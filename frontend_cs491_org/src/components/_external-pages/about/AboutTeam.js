import PropTypes from 'prop-types'
import { useRef } from 'react'
import Slider from 'react-slick'
import { Icon } from '@iconify/react'
import linkedinFill from '@iconify/icons-eva/linkedin-fill'
// material
import { useTheme } from '@material-ui/core/styles'
import { Box, Card, Button, Container, Typography, IconButton, Link } from '@material-ui/core'
// utils
import mockData from '../../../utils/mock-data'
//
import { varFadeIn, varFadeInUp, MotionInView } from '../../animate'
import { CarouselControlsArrowsBasic2 } from '../../carousel'

// ----------------------------------------------------------------------

const MOCK_MEMBERS = [...Array(4)].map((_, index) => ({
  id: mockData.id(index),
  name: mockData.name.fullName(index),
  role: mockData.role(index),
  avatar: mockData.image.avatar(index),
  link: mockData.socialProfile(index),
}))

// ----------------------------------------------------------------------

MemberCard.propTypes = {
  member: PropTypes.shape({
    id: PropTypes.string,
    avatar: PropTypes.string,
    name: PropTypes.string,
    role: PropTypes.string,
    link: PropTypes.string,
  }),
}

function MemberCard({ member }) {
  const { name, role, avatar, link } = member
  return (
    <Card key={name} sx={{ p: 1, mx: 1.5 }}>
      <Typography variant="subtitle1" sx={{ mt: 2, mb: 0.5 }}>
        {name}
      </Typography>
      <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
        {role}
      </Typography>
      <Box component="img" src={avatar} sx={{ width: '100%', borderRadius: 1.5 }} />
      <Box sx={{ mt: 2, mb: 1 }}>
        {[linkedinFill].map((social, index) => (
          <IconButton key={index} href={link} target="_blank">
            <Icon icon={social} width={20} height={20} href={link} target="_blank" />
            <Link href={link} target="_blank"></Link>
          </IconButton>
        ))}
      </Box>
    </Card>
  )
}

export default function AboutTeam() {
  const carouselRef = useRef()
  const theme = useTheme()

  const settings = {
    slidesToShow: 4,
    centerMode: true,
    centerPadding: '0 80px',
    rtl: Boolean(theme.direction === 'rtl'),
    responsive: [
      {
        breakpoint: 1279,
        settings: { slidesToShow: 3 },
      },
      {
        breakpoint: 959,
        settings: { slidesToShow: 2 },
      },
      {
        breakpoint: 600,
        settings: { slidesToShow: 1 },
      },
    ],
  }

  const handlePrevious = () => {
    carouselRef.current.slickPrev()
  }

  const handleNext = () => {
    carouselRef.current.slickNext()
  }

  return (
    <Container maxWidth="lg" sx={{ pb: 10, textAlign: 'center' }}>
      <MotionInView variants={varFadeInUp}>
        <Typography variant="h2" sx={{ mb: 3 }}>
          Our Team
        </Typography>
      </MotionInView>
      <Box sx={{ position: 'relative' }}>
        <Slider ref={carouselRef} {...settings}>
          {MOCK_MEMBERS.map((member) => (
            <MotionInView key={member.id} variants={varFadeIn}>
              <MemberCard member={member} />
            </MotionInView>
          ))}
        </Slider>
        <CarouselControlsArrowsBasic2
          onNext={handleNext}
          onPrevious={handlePrevious}
          sx={{ transform: 'translateY(-64px)' }}
        />
      </Box>
    </Container>
  )
}
