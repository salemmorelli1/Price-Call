/* Close an old static publication when its next XNYS session settles. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.PriceCallFreshness = api;
})(globalThis, function () {
  'use strict';
  function derive(snapshot, now = Date.now()) {
    const deadline = Date.parse(snapshot.freshness_expires_at_utc);
    if (Number.isFinite(deadline) && Number.isFinite(now) && now < deadline) {
      return snapshot;
    }
    const reason = 'The next XNYS session has settled or the publication expiry is unavailable; refresh the verified production snapshot.';
    const mode = snapshot.publish_mode === 'NORMAL' ? 'FAIL_CLOSED_STALE' : snapshot.publish_mode;
    const validation = snapshot.operator_validation || {};
    return {
      ...snapshot,
      final_pass: false,
      data_freshness_ok: false,
      publication_freshness_ok: false,
      publish_mode: mode,
      deployment_mode: 'NO_ACTION_STALE',
      operator_validation: {
        ...validation,
        status: 'NOT_VALIDATED',
        reasons: [...(validation.reasons || []).filter(x => x !== reason), reason]
      },
      lineage: {...(snapshot.lineage || {}), prediction_tape_paused: true, message: reason}
    };
  }
  return {derive};
});
