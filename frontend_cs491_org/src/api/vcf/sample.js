import { useVcfFile } from './get'

export const useSampleName = ({ fileId }) => {
  const vcfFile = useVcfFile({ fileId })

  const data = vcfFile.data?.sample
  return { ...vcfFile, data }
}
