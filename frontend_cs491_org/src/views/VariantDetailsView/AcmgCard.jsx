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

function AcmgCard({ variant, height }) {
  const [acmgInterpretation, acmgCriterias] = variant?.ACMG ?? ['', []]

  const acmgSeverity = ACMG_SEVERITY[acmgInterpretation]

  return (
    <Card sx={{ height }}>
      {variant === undefined ? (
        <Stack direction="column" justifyContent="center" alignItems="center" sx={{ height: '100%' }}>
          <CircularProgress size={150} />
        </Stack>
      ) : (
        <Stack direction="row" alignItems="center" justifyContent="center" spacing={3} sx={{ width: 1, height }}>
          <Stack direction="column" alignItems="center" justifyContent="center">
            <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
              {acmgCriterias.map((criteria) => (
                <Label key={criteria} color="default" variant="ghost" sx={{ px: 1, py: 2 }}>
                  <Typography variant="button">{criteria}</Typography>
                </Label>
              ))}
            </Stack>
            <Typography variant="body2" sx={{ opacity: 0.72 }}>
              ACMG Criterias
            </Typography>
          </Stack>
        </Stack>
      )}
    </Card>
  )
}

export default AcmgCard
