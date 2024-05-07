// tableUtils.js

/**
 * Sorts an array of objects based on a key and direction.
 * @param {Array} data - The data array to be sorted.
 * @param {String} key - The key to sort by.
 * @param {String} direction - 'ascending' or 'descending'.
 * @returns {Array} - The sorted array.
 */
export const sortRows = (data, key, direction) => {
    return data.sort((a, b) => {
        if (a[key] < b[key]) {
            return direction === 'ascending' ? -1 : 1;
        }
        if (a[key] > b[key]) {
            return direction === 'ascending' ? 1 : -1;
        }
        return 0;
    });
};

/**
 * Filters an array of objects based on given configuration.
 * @param {Array} data - The data array to be filtered.
 * @param {Object} config - The filter configuration.
 * @returns {Array} - The filtered array.
 */
export const filterRows = (data, config) => {
    return data.filter(row => {
        //const freqInRange = row.frequency >= config.freqRange[0] && row.frequency <= config.freqRange[1];
        const scoreInRange = row.priovar_score >= config.scoreRange[0] && row.priovar_score <= config.scoreRange[1];
        //const gtMatch = config.gts.length === 0 || config.gts.includes(row.gt);
        const acmgScoreMatch = config.strengths.length ? config.strengths.includes(row.acmgScore) : true;

        //return freqInRange && scoreInRange && gtMatch && acmgScoreMatch;
        console.log("scoreInRange: ", scoreInRange)
        return scoreInRange && acmgScoreMatch;
    });
};
