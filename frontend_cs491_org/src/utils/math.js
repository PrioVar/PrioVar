// https://stackoverflow.com/a/46171960
function toFixed(x) {
  if (Math.abs(x) < 1.0) {
    var e = parseInt(x.toString().split('e-')[1])
    if (e) {
      x *= 10 ** (e - 1)
      x = `0.${new Array(e).join('0')}${x.toString().substring(2)}`
    }
  } else {
    var e = parseInt(x.toString().split('+')[1])
    if (e > 20) {
      e -= 20
      x /= 10 ** e
      x += new Array(e + 1).join('0')
    }
  }
  return x
}

export function toFixedTrunc(x, n) {
  x = toFixed(`${x}`)

  // From here on the code is the same than the original answer
  const v = `${x}`.split('.')
  if (n <= 0) return v[0]
  let f = v[1] || ''
  if (f.length > n) return `${v[0]}.${f.substr(0, n)}`
  while (f.length < n) f += '0'
  const result = `${v[0]}.${f}`

  return result === 'null.00' ? 'NaN' : result
}

export function abbreviateNumber(number) {
  const SI_SYMBOL = ['', 'k', 'M', 'B', 'T', 'P', 'E']

  // what tier? (determines SI symbol)
  const tier = (Math.log10(number) / 3) | 0

  // if zero, we don't need a suffix
  if (tier === 0) return number

  // get suffix and determine scale
  const suffix = SI_SYMBOL[tier]
  const scale = 10 ** (tier * 3)

  // scale the number
  const scaled = number / scale

  // format number and add suffix
  return scaled.toFixed(1) + suffix
}
