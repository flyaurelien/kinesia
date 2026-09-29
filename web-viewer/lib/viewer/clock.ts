/**
 * Playback clock shared by the 3D stage, the video and the timeline.
 *
 * While the video can play, it is the master: its `currentTime` is what the
 * user sees, so the 3D scene follows it rather than a separate timer that
 * would drift. Without a playable video an internal clock takes over.
 */

type Listener = () => void;

export class PlaybackClock {
  readonly fps: number;
  readonly duration: number;
  private video: HTMLVideoElement | null = null;
  private internalTime = 0;
  private internalStarted: number | null = null;
  private playingInternal = false;
  private rateValue = 1;
  private listeners = new Set<Listener>();

  constructor(fps: number, frames: number) {
    this.fps = fps;
    this.duration = frames / fps;
  }

  attach(video: HTMLVideoElement | null): void {
    this.video = video;
    if (video) {
      video.playbackRate = this.rateValue;
      video.currentTime = this.internalTime;
    }
  }

  private usable(): HTMLVideoElement | null {
    const v = this.video;
    return v && v.readyState >= 1 && !v.error ? v : null;
  }

  /** Current time in seconds. */
  now(): number {
    const v = this.usable();
    if (v) return v.currentTime;
    if (this.playingInternal && this.internalStarted !== null) {
      const t = this.internalTime + ((performance.now() - this.internalStarted) / 1000) * this.rateValue;
      if (t >= this.duration) {
        this.internalTime = this.duration;
        this.playingInternal = false;
        this.internalStarted = null;
        this.emit();
        return this.duration;
      }
      return t;
    }
    return this.internalTime;
  }

  /** Current (fractional) frame index, clamped to the clip. */
  frame(): number {
    return Math.min(Math.max(0, this.now() * this.fps), Math.max(0, this.duration * this.fps - 1));
  }

  get playing(): boolean {
    const v = this.usable();
    return v ? !v.paused && !v.ended : this.playingInternal;
  }

  get rate(): number {
    return this.rateValue;
  }

  play(): void {
    const v = this.usable();
    if (v) {
      if (v.ended || v.currentTime >= this.duration - 1e-3) v.currentTime = 0;
      void v.play().catch(() => undefined);
    } else {
      if (this.internalTime >= this.duration) this.internalTime = 0;
      this.playingInternal = true;
      this.internalStarted = performance.now();
    }
    this.emit();
  }

  pause(): void {
    const v = this.usable();
    if (v) v.pause();
    this.internalTime = this.now();
    this.playingInternal = false;
    this.internalStarted = null;
    this.emit();
  }

  toggle(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  seek(seconds: number): void {
    const t = Math.min(Math.max(0, seconds), this.duration);
    this.internalTime = t;
    if (this.internalStarted !== null) this.internalStarted = performance.now();
    const v = this.usable() ?? this.video;
    if (v) v.currentTime = t;
    this.emit();
  }

  /** Move by whole frames, landing in the middle of a frame (pauses playback). */
  step(frames: number): void {
    this.pause();
    // The frame on screen is floor(frame()); rounding would read a frame's
    // middle as the next frame.
    const target = Math.floor(this.frame()) + frames;
    this.seek((target + 0.5) / this.fps);
  }

  setRate(rate: number): void {
    this.rateValue = rate;
    const v = this.video;
    if (v) v.playbackRate = rate;
    if (this.playingInternal) {
      this.internalTime = this.now();
      this.internalStarted = performance.now();
    }
    this.emit();
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  emit(): void {
    for (const listener of this.listeners) listener();
  }
}
