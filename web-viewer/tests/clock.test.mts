/** Frame stepping must move exactly one frame each way, from anywhere in a frame. */
import assert from "node:assert/strict";
import { test } from "node:test";

import { PlaybackClock } from "../lib/viewer/clock.ts";

for (const fps of [23.976, 24, 25, 29.97, 30, 59.94]) {
  test(`step moves one frame at a time at ${fps} fps`, () => {
    const clock = new PlaybackClock(fps, 400);
    clock.seek(10.3 / fps); // somewhere inside frame 10
    for (let expected = 11; expected <= 60; expected++) {
      clock.step(1);
      assert.equal(Math.floor(clock.frame()), expected);
    }
    for (let expected = 59; expected >= 0; expected--) {
      clock.step(-1);
      assert.equal(Math.floor(clock.frame()), expected);
    }
    clock.step(-1);
    assert.equal(Math.floor(clock.frame()), 0);
  });
}
