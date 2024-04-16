import { Autocomplete, Box, CircularProgress, TextField } from '@material-ui/core'
import MUIDataTable from 'mui-datatables'
import { chain as flatMap } from 'ramda'
import React, { useState } from 'react'
import SampleSelector from 'src/components/Medical/SampleSelector'
import useLazyEffect from 'src/hooks/useLazyEffect'
import { updateTrio, useFiles } from '../../api/vcf'

const fixResults = (rows) => {
  return rows.map(({ name, ...rest }) => {
    return {
      name: name.replace('vcf_files/', ''),
      ...rest,
    }
  })
}

const TrioSelector = function ({ options, trio, samples, fileId }) {
  const [motherOption, setMotherOption] = useState({
    label: trio.mother_sample_name,
    value: { fileId: trio.mother_file, sample: trio.mother_sample_name },
  })
  const [fatherOption, setFatherOption] = useState({
    label: trio.father_sample_name,
    value: { fileId: trio.father_file, sample: trio.father_sample_name },
  })

  useLazyEffect(() => {
    updateTrio(fileId, { mother: motherOption.value, father: fatherOption.value })
  }, [fileId, motherOption, fatherOption])

  return (
    <Box display="flex" flexDirection="row">
      <Autocomplete
        options={options.filter(
          (o) =>
            ![trio.mother_sample_name, trio.father_sample_name, ...samples, fatherOption.value.sample].includes(
              o.value.sample,
            ),
        )}
        style={{ width: 200 }}
        renderInput={(params) => <TextField {...params} label="Mother" variant="outlined" />}
        value={motherOption}
        getOptionLabel={(option) => option.label || ''}
        onChange={(_e, newMother) => {
          newMother ||= {
            label: '',
            value: { fileId: undefined, sample: undefined },
          }

          setMotherOption(newMother)
        }}
      />
      <Box p={1} />
      <Autocomplete
        options={options.filter(
          (o) =>
            ![trio.mother_sample_name, trio.father_sample_name, ...samples, motherOption.value.sample].includes(
              o.value.sample,
            ),
        )}
        style={{ width: 200 }}
        renderInput={(params) => <TextField {...params} label="Father" variant="outlined" />}
        value={fatherOption}
        getOptionLabel={(option) => option.label || ''}
        onChange={(_e, newFather) => {
          newFather ||= {
            label: '',
            value: { fileId: undefined, sample: undefined },
          }

          setFatherOption(newFather)
        }}
      />
    </Box>
  )
}

const createTrioOptions = (rows) => {
  return flatMap(
    (row) =>
      row.samples.map((s) => ({
        label: s,
        value: { fileId: row.id, sample: s },
      })),
    rows,
  )
}

const SamplesView = function () {
  const { status, data = [] } = useFiles().query

  const trioOptions = createTrioOptions(data)

  const COLUMNS = [
    {
      name: 'name',
      label: 'Filename 📂',
      options: {
        filter: true,
        sort: true,
      },
    },
    {
      name: 'samples',
      label: 'Samples',
      options: {
        filter: true,
        sort: true,
        customBodyRenderLite: (dataIndex) => {
          const row = data[dataIndex]
          if (!row) {
            return null
          }

          return <SampleSelector fileId={row.id} />
        },
      },
    },
    {
      name: 'trio',
      label: 'Trio',
      options: {
        filter: true,
        sort: true,
        customBodyRenderLite: (dataIndex) => {
          const row = data[dataIndex]
          if (!row) {
            return null
          }

          return <TrioSelector trio={row.trio} options={trioOptions} samples={row.samples} fileId={row.id} />
        },
      },
    },
  ]

  switch (status) {
    case 'success':
      return <MUIDataTable title="Samples" data={fixResults(data)} columns={COLUMNS} />
    default:
      return <CircularProgress />
  }
}

export default SamplesView
