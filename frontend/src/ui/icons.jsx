// Small SF-Symbols-style line icons. currentColor, 24px grid.
const Icon = ({ children, size = 22, stroke = 1.8, ...rest }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={stroke}
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...rest}>{children}</svg>
)

export const Compass = (p) => <Icon {...p}><circle cx="12" cy="12" r="9" /><path d="m15.5 8.5-2 5-5 2 2-5z" /></Icon>
export const Scan = (p) => <Icon {...p}><path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2" /><path d="M8 9h8M8 12h8M8 15h5" /></Icon>
export const Palate = (p) => <Icon {...p}><path d="M12 3v18M3 12h18M5.6 5.6l12.8 12.8M18.4 5.6 5.6 18.4" opacity=".35" /><path d="M12 5.5 16.6 9l1.2 5-4.3 3.6-4.9-.9L6.4 11l1.9-4.4z" /></Icon>
export const Heart = ({ filled, ...p }) => <Icon {...p} fill={filled ? 'currentColor' : 'none'}><path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z" /></Icon>
export const Star = ({ filled, ...p }) => <Icon {...p} fill={filled ? 'currentColor' : 'none'}><path d="m12 3.5 2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z" /></Icon>
export const Chevron = (p) => <Icon {...p}><path d="m9 6 6 6-6 6" /></Icon>
export const Back = (p) => <Icon {...p}><path d="m15 6-6 6 6 6" /></Icon>
export const Camera = (p) => <Icon {...p}><path d="M4 8h3l2-3h6l2 3h3v11H4z" /><circle cx="12" cy="13" r="3.5" /></Icon>
export const Upload = (p) => <Icon {...p}><path d="M12 15V4M7.5 8.5 12 4l4.5 4.5M5 15v4h14v-4" /></Icon>
export const Share = (p) => <Icon {...p}><path d="M12 15V4M8 7.5 12 3.5l4 4M6 11H5v9h14v-9h-1" /></Icon>
export const Sparkle = (p) => <Icon {...p}><path d="M12 3.5c.6 4.2 2.3 5.9 6.5 6.5-4.2.6-5.9 2.3-6.5 6.5-.6-4.2-2.3-5.9-6.5-6.5 4.2-.6 5.9-2.3 6.5-6.5zM18.5 15.5c.3 1.6.9 2.2 2.5 2.5-1.6.3-2.2.9-2.5 2.5-.3-1.6-.9-2.2-2.5-2.5 1.6-.3 2.2-.9 2.5-2.5z" /></Icon>
export const Alert = (p) => <Icon {...p}><path d="M12 4 3 19h18z" /><path d="M12 10v4M12 16.5v.5" /></Icon>
export const Check = (p) => <Icon {...p}><path d="m5 12.5 4.5 4.5L19 7.5" /></Icon>
export const Close = (p) => <Icon {...p}><path d="M6 6l12 12M18 6 6 18" /></Icon>
export const Receipt = (p) => <Icon {...p}><path d="M6 3h12v18l-3-2-3 2-3-2-3 2z" /><path d="M9 8h6M9 12h6" /></Icon>
export const Plus = (p) => <Icon {...p}><path d="M12 5v14M5 12h14" /></Icon>
export const Pin = (p) => <Icon {...p}><path d="M12 21s-6.5-5.7-6.5-11a6.5 6.5 0 0 1 13 0c0 5.3-6.5 11-6.5 11z" /><circle cx="12" cy="10" r="2.3" /></Icon>
export const Cloud = (p) => <Icon {...p}><path d="M7 18h10a4 4 0 0 0 .5-8A6 6 0 0 0 6 9.5 4.3 4.3 0 0 0 7 18z" /></Icon>
export const Leaf = (p) => <Icon {...p}><path d="M5 19c0-8 5-13 14-14-1 9-6 14-14 14zM5 19l7-7" /></Icon>

// iOS status bar glyphs (filled)
export const StatusGlyphs = () => (
  <svg width="68" height="12" viewBox="0 0 68 12" aria-hidden="true" fill="currentColor">
    <rect x="0" y="8" width="3" height="4" rx="1" /><rect x="4.5" y="6" width="3" height="6" rx="1" />
    <rect x="9" y="3.5" width="3" height="8.5" rx="1" /><rect x="13.5" y="1" width="3" height="11" rx="1" />
    <path d="M30 3.2a9.5 9.5 0 0 1 12.8 0l-1.3 1.4a7.6 7.6 0 0 0-10.2 0zM32.4 5.9a6 6 0 0 1 8 0l-1.3 1.4a4.1 4.1 0 0 0-5.4 0zM34.8 8.6a2.5 2.5 0 0 1 3.2 0L36.4 10.3z" />
    <rect x="46.5" y="1" width="18" height="10" rx="3" fill="none" stroke="currentColor" strokeOpacity=".4" />
    <rect x="48" y="2.5" width="13.5" height="7" rx="1.6" /><rect x="65.5" y="4" width="1.6" height="4" rx=".8" fillOpacity=".4" />
  </svg>
)
