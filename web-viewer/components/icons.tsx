/** Small inline icons (24x24 grid, stroke = currentColor). */

import type { SVGProps } from "react";

type Props = SVGProps<SVGSVGElement> & { size?: number };

function Icon({ size = 16, children, ...rest }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      {children}
    </svg>
  );
}

export const Plus = (p: Props) => <Icon {...p}><path d="M12 5v14M5 12h14" /></Icon>;
export const Upload = (p: Props) => <Icon {...p}><path d="M12 16V4M7 9l5-5 5 5M5 20h14" /></Icon>;
export const Play = (p: Props) => <Icon {...p}><path d="M7 5l12 7-12 7z" fill="currentColor" stroke="none" /></Icon>;
export const Pause = (p: Props) => <Icon {...p}><path d="M8 5v14M16 5v14" strokeWidth={3} /></Icon>;
export const StepBack = (p: Props) => <Icon {...p}><path d="M18 6l-8 6 8 6zM6 6v12" /></Icon>;
export const StepForward = (p: Props) => <Icon {...p}><path d="M6 6l8 6-8 6zM18 6v12" /></Icon>;
export const Eye = (p: Props) => <Icon {...p}><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></Icon>;
export const EyeOff = (p: Props) => <Icon {...p}><path d="M3 3l18 18M10.6 5.1A10 10 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-3.2 4.1M6.6 6.6C3.9 8.4 2 12 2 12s3.5 7 10 7a9.7 9.7 0 0 0 4.4-1" /></Icon>;
export const Target = (p: Props) => <Icon {...p}><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="3" /></Icon>;
export const Orbit = (p: Props) => <Icon {...p}><ellipse cx="12" cy="12" rx="9" ry="4.5" /><circle cx="12" cy="12" r="2.2" fill="currentColor" stroke="none" /></Icon>;
export const Camera = (p: Props) => <Icon {...p}><path d="M3 8h4l2-3h6l2 3h4v11H3z" /><circle cx="12" cy="13" r="3.5" /></Icon>;
export const Grid = (p: Props) => <Icon {...p}><rect x="4" y="4" width="16" height="16" rx="1" /><path d="M4 12h16M12 4v16" /></Icon>;
export const User = (p: Props) => <Icon {...p}><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></Icon>;
export const Download = (p: Props) => <Icon {...p}><path d="M12 4v12M7 11l5 5 5-5M5 20h14" /></Icon>;
export const Trash = (p: Props) => <Icon {...p}><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" /></Icon>;
export const Back = (p: Props) => <Icon {...p}><path d="M15 5l-7 7 7 7" /></Icon>;
export const Close = (p: Props) => <Icon {...p}><path d="M6 6l12 12M18 6L6 18" /></Icon>;
export const Refresh = (p: Props) => <Icon {...p}><path d="M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6" /></Icon>;
export const Stop = (p: Props) => <Icon {...p}><rect x="6" y="6" width="12" height="12" rx="1.5" /></Icon>;
export const Check = (p: Props) => <Icon {...p}><path d="M5 12l5 5 9-10" /></Icon>;
export const Chip = (p: Props) => <Icon {...p}><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" /></Icon>;
export const Video = (p: Props) => <Icon {...p}><rect x="3" y="6" width="13" height="12" rx="2" /><path d="M16 10l5-3v10l-5-3" /></Icon>;
export const Bones = (p: Props) => <Icon {...p}><circle cx="12" cy="5" r="2" /><path d="M12 7v7M12 14l-4 6M12 14l4 6M6 10l6-2 6 2" /></Icon>;
export const Body = (p: Props) => <Icon {...p}><circle cx="12" cy="4.5" r="2.2" /><path d="M8 9h8l-1 6h-2l-.5 6h-1L11 15H9z" /></Icon>;
export const Trail = (p: Props) => <Icon {...p}><path d="M4 18c3 0 3-6 6-6s3 4 6 4 3-8 5-8" strokeDasharray="2 3" /></Icon>;
export const Tag = (p: Props) => <Icon {...p}><path d="M3 12V4h8l10 10-8 8z" /><circle cx="7.5" cy="7.5" r="1.3" fill="currentColor" /></Icon>;
export const Maximize = (p: Props) => <Icon {...p}><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" /></Icon>;
export const Swap = (p: Props) => <Icon {...p}><path d="M7 7h11l-3-3M17 17H6l3 3" /></Icon>;
