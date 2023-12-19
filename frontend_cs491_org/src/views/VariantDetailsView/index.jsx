import { Box, CircularProgress, Grid, Stack } from '@material-ui/core'
import React from 'react'
import { useParams } from 'react-router-dom'
import Page from 'src/components/Page'
import DatabasesCard from 'src/views/VariantDetailsView/DatabasesCard'
import FrequencyCard from 'src/views/VariantDetailsView/FrequencyCard'
import GenesTabSection from 'src/views/VariantDetailsView/GenesTab'
import HgvsCard from 'src/views/VariantDetailsView/HgvsCard'
import { GeneSmybolsCard, ImpactsCard } from 'src/views/VariantDetailsView/MiscCard'
import LocationCard from 'src/views/VariantDetailsView/LocationCard'
import NotesCard from 'src/views/VariantDetailsView/Notes'
import PathogenicityCard from 'src/views/VariantDetailsView/PathogenicityCard'
import ReadDetailsCard from 'src/views/VariantDetailsView/ReadDetailsCard'
import SpliceCard from 'src/views/VariantDetailsView/SpliceCard'
import TranscriptFactorsTab from 'src/views/VariantDetailsView/TranscriptFactorsTab'
import VariantTab from 'src/views/VariantDetailsView/VariantTab'
import Igv from './Igv'
import Section from './Section'
import { useVariantDetails } from '../../api/variant'
import MatchmakingCard from './MatchmakingCard'
import AcmgCard from './AcmgCard'
import ReactDOM from 'react-dom'
import NavbarRoutes from '../VariantsView/NavbarRoutes'
import { VAR_DETAILS_NAVBAR_CONFIG } from 'src/constants/index'
import { TableView } from '@material-ui/icons'

const Loading = function () {
  return (
    <Stack direction="row" justifyContent="center">
      <CircularProgress size="10vh" />
    </Stack>
  )
}

const VariantDetailsView = function () {
  const { fileId, sampleName, chrom, pos } = useParams()
  const { status, data = {} } = useVariantDetails({ fileId, sampleName, chrom, pos })

  const { variant, disease, omim, matchmaking } = data

  return (
    <Page title="Variant Details | PrioVar">
      {ReactDOM.createPortal(
        <Stack direction="row" spacing={2}>
          <NavbarRoutes navConfig={VAR_DETAILS_NAVBAR_CONFIG} params={{ fileId, sampleName }} />
        </Stack>,
        document.getElementById('custom-toolbar-container'),
      )}
      <Grid container spacing={2}>
        <Grid item xl={4} sm={6} xs={12}>
          <PathogenicityCard variant={variant} />
        </Grid>
        <Grid container item xl lg sm xs rowSpacing={6} columnSpacing={2}>
          <Grid item xl={7} xs={12}>
            <LocationCard variant={variant} height={150} />
          </Grid>
          <Grid item xl={5} xs={12}>
            <ReadDetailsCard variant={variant} height={150} />
          </Grid>
          <Grid item xl={6} xs={12} sx={{ display: { xl: 'unset', xs: 'none' } }}>
            <DatabasesCard variant={variant} height={150} />
          </Grid>
          <Grid item xl={3} xs={6} sx={{ display: { xl: 'unset', xs: 'none' } }}>
            <GeneSmybolsCard variant={variant} height={150} />
          </Grid>
          <Grid item xl={3} xs={6} sx={{ display: { xl: 'unset', xs: 'none' } }}>
            <AcmgCard variant={variant} height={150} />
          </Grid>
        </Grid>
        <Grid item xl={3} sm={6} xs={12}>
          <NotesCard variant={variant} height={310} sampleName={fileId} />
        </Grid>
      </Grid>
      <Box p={1} />
      <Grid container spacing={2}>
        <Grid item xl={6} xs={12} sx={{ display: { xl: 'none', xs: 'unset' } }}>
          <DatabasesCard variant={variant} height={150} />
        </Grid>
        <Grid item xl={6} xs={6} sx={{ display: { xl: 'none', xs: 'unset' } }}>
          <GeneSmybolsCard variant={variant} height={150} />
        </Grid>
        <Grid item xl={6} xs={6} sx={{ display: { xl: 'none', xs: 'unset' } }}>
          <AcmgCard variant={variant} height={200} />
        </Grid>
        <Grid item xl={2} sm={6} xs={12}>
          <FrequencyCard variant={variant} height={150} />
        </Grid>
        <Grid item xl={3} sm={6} xs={12}>
          <MatchmakingCard matchmaking={matchmaking} height={200} />
        </Grid>
        {/*<Grid item xl={3} sm={6} xs={12}>*/}
        {/*  <SpliceCard variant={variant} />*/}
        {/*</Grid>*/}
        <Grid item xl={4} sm={6} xs={12}>
          <HgvsCard variant={variant} />
        </Grid>
        <Grid item xl={3} sm={6} xs={12}>
          <ImpactsCard variant={variant} height={200} />
        </Grid>
      </Grid>
      {/* <Section title="Variant">{status === 'loading' ? <Loading /> : <VariantTab data={variant} />}</Section> */}
      <Box p={1} />
      {status === 'loading' ? (
        <Section title="Genes">
          <Loading />
        </Section>
      ) : (
        <GenesTabSection disease={disease} omim={omim} transcripts={variant.Transcripts} variantId={variant.ID} />
      )}
      <Box p={1} />
      <Section title="Transcripts">
        {status === 'loading' ? <Loading /> : <TranscriptFactorsTab data={variant.Transcripts} />}
      </Section>
      <Box p={1} />
      <Section title="Gene Browser">
        <Igv chrom={chrom} pos={parseInt(pos, 10)} />
      </Section>
    </Page>
  )
}

export default VariantDetailsView
