import igv from 'igv'
import PropTypes from 'prop-types'
import React, { useEffect, useRef } from 'react'

const Igv = function ({ chrom, pos }) {
  const igvRef = useRef()

  useEffect(() => {
    if (igvRef.current) {
      igv.createBrowser(igvRef.current, { genome: 'hg38', locus: `chr${chrom}:${pos}` })
      return () => {
        if (igv?.removeAllBrowsers) {
          igv.removeAllBrowsers()
        }
      }
    }
  }, [chrom, pos, igvRef])

  return <div id="igv-box" ref={igvRef} />
}

Igv.propTypes = {
  chrom: PropTypes.string.isRequired,
  pos: PropTypes.number.isRequired,
}

export default Igv
