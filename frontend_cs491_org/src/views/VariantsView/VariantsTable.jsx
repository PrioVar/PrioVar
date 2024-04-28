import { Box, IconButton, Tooltip, Typography } from '@material-ui/core'
//import FindInPageIcon from '@material-ui/icons/FindInPage'
import InfoIcon from '@material-ui/icons/Info'
//import TurnedInIcon from '@material-ui/icons/TurnedIn'
//import TurnedInNotIcon from '@material-ui/icons/TurnedInNot'
import WarningRoundedIcon from '@material-ui/icons/WarningRounded'
import { useTheme } from '@material-ui/styles'
import PropTypes from 'prop-types'
import React, { useCallback, useEffect, useMemo } from 'react'
import { useParams } from 'react-router-dom'
import ClickAwayTooltip from 'src/components/ClickAwayTooltip'
import { ACMG_SEVERITY, COLOR_ENCODE_DESCRIPTIONS } from 'src/constants'
import { getMostSevereColor, getMostSeverePvs1 } from 'src/utils/bio'
//import DatabasesCell from 'src/views/VariantsView/Cells/DatabasesCell'
import FrequencyCell from 'src/views/VariantsView/Cells/FrequencyCell'
import GeneSymbolsCell from 'src/views/VariantsView/Cells/GeneSymbolsCell'
//import GtCell from 'src/views/VariantsView/Cells/GtCell'
//import HgvsCCell from 'src/views/VariantsView/Cells/HgvsCCell'
//import HgvsPCell from 'src/views/VariantsView/Cells/HgvsPCell'
import PathogenicityCell from 'src/views/VariantsView/Cells/PathogenicityCell'
import PhenotypesCell from 'src/views/VariantsView/Cells/PhenotypesCell'
//import ReadDetailsCell from 'src/views/VariantsView/Cells/ReadDetailsCell'
import { useImmer } from 'use-immer'

import LibraTable from '../../components/LibraTable'
//import ImpactCell from './Cells/ImpactCell'
import { useSampleMetadata } from '../../api/vcf'
import { isHomogeneous } from 'src/utils/bio'

const INITIAL_DISPLAY_COLUMNS = [
  {
    keys: ['Libra Pathogenicity', 'ClinVar SIG', 'SIFT', 'Polyphen', 'CADD', 'REVEL', 'DANN', 'MetalR'],
    label: 'Pathogenicity',
    renderCell: ([libraP, clinVar, sift, polpyhen, cadd, revel, dann, metalR]) => (
      <PathogenicityCell
        clinVar={clinVar}
        sift={sift}
        polyphen={polpyhen}
        cadd={cadd}
        revel={revel}
        libraP={libraP}
        dann={dann}
        metalR={metalR}
      />
    ),
    collapse: true,
  },
  {
    keys: ['Gene Name'],
    label: 'Gene Symbol',
    renderCell: ([symbols]) => <GeneSymbolsCell symbols={symbols} />,
    collapse: true,
  },
  {
    keys: ['Phenotypes'],
    label: 'Related Diseases',
    renderCell: ([phenotypes]) => <PhenotypesCell phenotypes={phenotypes} />,
    width: 450,
    collapse: true,
  },
  {
    keys: ['AF', 'gnomAD AF', '1KG Frequency', 'TV AF', 'ExAC AF'],
    label: 'Frequency',
    renderCell: ([libraAf, gnomAdAf, oneKgAf, tvAf, exAcAf]) => (
      <FrequencyCell libraAf={libraAf} gnomAdAf={gnomAdAf} oneKgAf={oneKgAf} tvAf={tvAf} exAcAf={exAcAf} />
    ),
    collapse: true,
  },
  {
    keys: ['PrioVar Pathogenicity Score'],
    label: 'PrioVar Pathogenicity Score',
    renderCell: ([libraAf, gnomAdAf, oneKgAf, tvAf, exAcAf]) => (
      <FrequencyCell libraAf={libraAf} gnomAdAf={gnomAdAf} oneKgAf={oneKgAf} tvAf={tvAf} exAcAf={exAcAf} />
    ),
    collapse: true,
  },
]

const isRowSelected = (selectedVariants, chrom, pos) => {
  return !!selectedVariants.find((sv) => sv.chrom === chrom && sv.pos === pos)
}

const getCurrentTable = (tableId, sampleMetadata) => {
  const data = sampleMetadata.query.data?.tables ?? []
  return data.find(({ id }) => id === tableId)
}

const VariantsTable = function ({
  id,
  showOnlySelectedVariants = false,
  page,
  pageSize,
  sortOrder,
  setPage,
  setPageSize,
  setSortOrder,
  data,
  status,
}) {
  const theme = useTheme()
  const { fileId, sampleName } = useParams()
  const sampleMetadata = useSampleMetadata({ fileId, sampleName })
  const [displayColumns, setDisplayColumns] = useImmer([])

  const currentTable = getCurrentTable(id, sampleMetadata)

  const { rows = [], columns = [], count = 0 } = data

  useEffect(() => {
    if (columns.length > 0 && displayColumns.length === 0) {
      setDisplayColumns(() => INITIAL_DISPLAY_COLUMNS)
    }
  }, [columns, displayColumns.length, setDisplayColumns])

  const handleClickBookmark = useCallback(
    (rowChrom, rowPos) => {
      if (isRowSelected(currentTable.selectedVariants, rowChrom, rowPos)) {
        sampleMetadata.updateTable(id, {
          ...currentTable,
          selectedVariants: currentTable.selectedVariants.filter(
            ({ chrom, pos }) => chrom !== rowChrom || rowPos !== pos,
          ),
        })
      } else {
        sampleMetadata.updateTable(id, {
          ...currentTable,
          selectedVariants: [...currentTable.selectedVariants, { chrom: rowChrom, pos: rowPos }],
        })
      }
    },
    [currentTable, id, sampleMetadata],
  )

  const iconColumns = useMemo(
    () => [
      {
        // color encoding icon
        // needed keys: acmg, clinvar, gnomad, exac, 1kg, impact, omim, gt
        keys: ['ACMG', 'ClinVar SIG', 'gnomAD AF', 'ExAC AF', '1KG Frequency', 'Impact', 'OMIM', 'GT'],
        renderCell: ([acmg, clinvar, gnomad, exac, kg, impact, omim, gt]) => {
          const is_pathogenic = clinvar?.find((x) => x === 'pathogenic') !== undefined
          const is_severe = impact?.find((x) => x === 'stop_gained' || x === 'stop_lost') !== undefined
          const is_frameshift = impact?.find((x) => x === 'frameshift_variant') !== undefined
          const is_splicing =
            impact?.find((x) => x === 'splice_acceptor_variant' || x === 'splice_donor_variant') !== undefined
          const is_missense = impact?.find((x) => x === 'missense_variant') !== undefined
          const hom = gt && isHomogeneous(gt)

          const getEncoding = (
            omim,
            is_pathogenic,
            exac,
            kg,
            is_severe,
            is_frameshift,
            is_splicing,
            is_missense,
            hom,
          ) => {
            if (omim && is_pathogenic && exac <= 0.01 && kg <= 0.01) return ['#FFD500', COLOR_ENCODE_DESCRIPTIONS.OMIM]
            if (is_severe && hom) return ['#D63C36', COLOR_ENCODE_DESCRIPTIONS.SEVERE]
            if (is_severe && !hom) return ['#FA5849', COLOR_ENCODE_DESCRIPTIONS.SEVERE]
            if (is_frameshift && hom) return ['#E46F57', COLOR_ENCODE_DESCRIPTIONS.FRAMESHIFT]
            if (is_frameshift && !hom) return ['#FA7A5F', COLOR_ENCODE_DESCRIPTIONS.FRAMESHIFT]
            if (is_splicing && hom) return ['#E49877', COLOR_ENCODE_DESCRIPTIONS.SPLICE]
            if (is_splicing && !hom) return ['#FBA580', COLOR_ENCODE_DESCRIPTIONS.SPLICE]
            if (is_missense && hom) return ['#E1BB9A', COLOR_ENCODE_DESCRIPTIONS.MISSENSE]
            if (is_missense && !hom) return ['#FCD2AC', COLOR_ENCODE_DESCRIPTIONS.MISSENSE]
            return 'inherit'
          }
          const [color, description] = getEncoding(
            omim,
            is_pathogenic,
            exac,
            kg,
            is_severe,
            is_frameshift,
            is_splicing,
            is_missense,
            hom,
          )
          return (
            <Tooltip title={`${description}`} arrow>
              <span
                style={{
                  height: '25px',
                  width: '25px',
                  backgroundColor: `${color}`,
                  borderRadius: '50%',
                  display: 'inline-block',
                }}
              ></span>
            </Tooltip>
          )
        },
      },
      {
        keys: ['FILTER'],
        renderCell: ([filter]) => {
          if (!filter) {
            return null
          }

          const title = (
            <Typography variant="body1" sx={{ p: 0.5 }}>
              FILTER: {filter}
            </Typography>
          )
          return (
            <ClickAwayTooltip title={title} arrow>
              <Box display="flex" pr={1}>
                <WarningRoundedIcon />
              </Box>
            </ClickAwayTooltip>
          )
        },
      },
      {
        keys: ['PVS1', 'ACMG'],
        renderCell: ([pvs1, acmg]) => {
          if (!acmg) {
            return null
          }

          const [acmgInterpretation, acmgCriterias] = acmg

          const acmgSeverity = ACMG_SEVERITY[acmgInterpretation]
          const mostSeverePvs1 = pvs1 ? getMostSeverePvs1(pvs1) : undefined

          const mostSevereColor = getMostSevereColor(
            [mostSeverePvs1?.severity?.color, acmgSeverity.color].filter(Boolean),
          )

          if (mostSevereColor === 'default') {
            return null
          }

          const title = (
            <>
              {acmgSeverity.color !== 'default' && (
                <Typography variant="caption" sx={{ p: 0.5 }}>
                  ACMG Rule: {acmgCriterias.sort().join(',')}
                  <br /> Strength: {acmgSeverity.description}
                </Typography>
              )}
              {pvs1 && (
                <Typography variant="caption" sx={{ p: 0.5 }}>
                  PVS1 Rule: {pvs1.join(',')}
                  <br /> Strength: {mostSeverePvs1.severity.description}
                </Typography>
              )}
            </>
          )

          return (
            <Box
              display="flex"
              m={1}
              sx={{ border: `2px solid ${theme.palette[mostSevereColor].main}`, borderRadius: '10%' }}
            >
              {title}
            </Box>
          )
        },
      },
      {
        keys: ['CHROM', 'POS', 'REF', 'ALT'],
        renderCell: ([chrom, pos, ref, alt]) => {
          const title = (
            <Typography variant="body1" sx={{ p: 0.5 }}>
              Location: chr{chrom}-{pos}
              <br />
              Ref: {ref}&nbsp;&nbsp;Alt: {alt}
            </Typography>
          )

          return (
            <ClickAwayTooltip title={title} placement="left" interactive>
              <IconButton>
                <InfoIcon />
              </IconButton>
            </ClickAwayTooltip>
          )
        },
      },
    ],
    [currentTable.selectedVariants, fileId, handleClickBookmark, sampleName],
  )

  // FIXME: there is an animation bug, set transitionDuration={2000} to understand the backdrop is broken while closing
  return (
    <LibraTable
      rows={rows}
      page={page}
      pageSize={pageSize}
      onChangePage={setPage}
      onChangePageSize={setPageSize}
      onChangeSort={setSortOrder}
      sort={sortOrder}
      pageSizeOptions={[10, 15, 25, 50, 100]}
      totalRowCount={count}
      isLoading={status !== 'success'}
      isPaginated={!showOnlySelectedVariants}
      iconColumns={iconColumns}
      columnConfig={displayColumns}
      columns={columns}
    />
  )
}

VariantsTable.propTypes = {
  id: PropTypes.string.isRequired,
  defaultTitle: PropTypes.string.isRequired,
  readonlyTitle: PropTypes.bool.isRequired,
  variant: PropTypes.oneOf(['standard', 'kh1', 'kh2', 'kh3', 'kh4', 'kh5', 'kh6']),
  showOnlySelectedVariants: PropTypes.bool.isRequired,
  hpo: PropTypes.shape({
    similar: PropTypes.bool,
    ids: PropTypes.arrayOf(PropTypes.string),
    mode: PropTypes.oneOf(['all', 'any', 'most']),
  }),
}

export default VariantsTable

/*
const handleColumnHide = (_column, columnIndex) => {
    setDisplayColumns((draft) => {
      draft[columnIndex].hidden = !draft[columnIndex].hidden
    })
  }

  const handleColumnReorder = (sourceIndex, destinationIndex) => {
    setDisplayColumns((draft) => {
      // Remove and add the column at specified index
      const [removedColumn] = draft.splice(sourceIndex, 1)
      draft.splice(destinationIndex, 0, removedColumn)
    })
  }
*/