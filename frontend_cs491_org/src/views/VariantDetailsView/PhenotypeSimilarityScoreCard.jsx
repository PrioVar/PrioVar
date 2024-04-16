import { styled, alpha } from '@material-ui/core/styles'
import { useTheme } from '@material-ui/styles'

import React, { useMemo } from 'react'
import {
  Card,
  Typography,
  Stack,
  Box,
  CircularProgress,
  Paper,
  CardHeader,
  Tooltip,
  TextField,
  CardContent,
} from '@material-ui/core'
import { ResponsiveBar } from '@nivo/bar'
import { fShortenNumber } from 'src/utils/formatNumber'
import FrequencyCell from '../VariantsView/Cells/FrequencyCell'
import { ACMG_SEVERITY } from '../../constants'
import { getMostSevereColor, getMostSeverePvs1 } from '../../utils/bio'
import Label from '../../components/Label'

function PhenotypeSimilarityScoreCard({ variant, height }) {
  const [acmgInterpretation, acmgCriterias] = variant?.ACMG ?? ['', []]

  const acmgSeverity = ACMG_SEVERITY[acmgInterpretation]

  return (
    <Card sx={{ height }}>
      <CardHeader title="Phenotype Similarity" />
      <CardContent>
        {variant === undefined ? (
          <Stack direction="column" justifyContent="center" alignItems="center" sx={{ height: '100%' }}>
            <CircularProgress size={150} />
          </Stack>
        ) : (
          acmgCriterias.map((criteria) => (
            <Label key={criteria} color="default" variant="ghost" sx={{ fontSize: 13, p: 1 }}>
              {criteria}
            </Label>
          ))
        )}
      </CardContent>
    </Card>
  )
}

export default PhenotypeSimilarityScoreCard
