/**
 * Exponential moving average: smoothing to give progressively lower weights to older values.
 */
class EMA {
  constructor(smoothing) {
    this.alpha = smoothing
    this.lastValue = null
    this.callCount = 0
  }

  /**
   * @param {number} value
   * @returns {number}
   */
  update(value) {
    const beta = 1 - this.alpha
    this.lastValue = this.alpha * value + beta * this.lastValue
    this.callCount += 1
    return this.lastValue / (1 - beta ** this.callCount)
  }
}

export default EMA
