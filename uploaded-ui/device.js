// Browser adapters for the device helpers referenced by the supplied UI.
window.PalatableDevice = {
  native: false,
  async exportFile(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement('a'), {href: url, download: name});
    a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  },
  async shareText(text) {
    if (!navigator.share) return false;
    await navigator.share({text}); return true;
  }
};
// The upload did not include the delivery-history importer. Keep its entry point usable.
window.PalatableHistory = {render: () => '<h2>Delivery history</h2><p class="sub">The delivery-history importer was not included in these UI files. You can build your palate with the taste survey.</p><button class="primary" onclick="go(\'tolerances\')">Start taste survey →</button>'};
