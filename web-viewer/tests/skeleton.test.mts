/** The browser's forward kinematics must match the Python scene builder exactly. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { forwardKinematics, slerp } from "../lib/scene/skeleton.ts";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/kinematics.json", import.meta.url), "utf8"));

function close(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance: number) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) < tolerance, `index ${i}: ${actual[i]} vs ${expected[i]}`);
  }
}

test("forward kinematics matches kinesia.scene.quat.compose", () => {
  const out = new Float64Array(fixture.expected.length);
  forwardKinematics(fixture.parents, 1, fixture.pelvis, Float64Array.from(fixture.local), out);
  close(out, fixture.expected, 1e-9);
});

test("slerp matches kinesia.scene.quat.slerp", () => {
  const out = new Float64Array(4);
  slerp(fixture.slerp.a, 0, fixture.slerp.b, 0, fixture.slerp.t, out, 0);
  const dot = out.reduce((sum, v, i) => sum + v * fixture.slerp.expected[i], 0);
  assert.ok(Math.abs(Math.abs(dot) - 1) < 1e-9);
});
