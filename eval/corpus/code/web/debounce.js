// debounce.js - delay calling fn until it has stopped being called for `wait` ms.
// Used for the search box so we don't hit the API on every keystroke.

/**
 * @param {Function} fn
 * @param {number} wait milliseconds of quiet before firing
 * @param {{leading?: boolean}} [options] leading: also fire on the first call of a burst
 * @returns {Function & {cancel: Function, flush: Function}}
 */
export function debounce(fn, wait = 300, options = {}) {
  const { leading = false } = options;
  let timer = null;
  let lastArgs = null;
  let lastThis = null;

  function invoke() {
    const args = lastArgs;
    const ctx = lastThis;
    lastArgs = null;
    lastThis = null;
    return fn.apply(ctx, args);
  }

  function debounced(...args) {
    lastArgs = args;
    lastThis = this;

    if (timer === null && leading) {
      // first call in a burst fires right away, and clears the pending args
      invoke();
    }

    clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      // trailing call only if something arrived since the last invoke
      if (lastArgs !== null) invoke();
    }, wait);
  }

  debounced.cancel = () => {
    clearTimeout(timer);
    timer = null;
    lastArgs = null;
    lastThis = null;
  };

  debounced.flush = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
      if (lastArgs !== null) return invoke();
    }
  };

  return debounced;
}

export default debounce;
