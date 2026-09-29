"use client";

import { Grid, Html, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import type { LoadedScene } from "@/lib/scene/load";
import type { PlaybackClock } from "@/lib/viewer/clock";
import { PersonObject, SharedBody } from "@/lib/viewer/person-object";

export type CameraMode = "orbit" | "follow" | "top" | "camera";

export type Display = { mesh: boolean; skeleton: boolean; trails: boolean; labels: boolean };

type Props = {
  data: LoadedScene;
  clock: PlaybackClock;
  hidden: Set<number>;
  selected: number | null;
  solo: boolean;
  display: Display;
  cameraMode: CameraMode;
  labels: Record<number, string>;
  video: HTMLVideoElement | null;
  onSelect: (id: number | null) => void;
};

/** World (z up, metres) to three.js (y up). */
const toThree = (x: number, y: number, z: number, out = new THREE.Vector3()) => out.set(x, z, -y);
const WORLD_TO_THREE = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const CAMERA_AXES = new THREE.Quaternion(1, 0, 0, 0); // camera y-down/z-forward to three's y-up/z-back

/** Where people spend their time: brief glimpses far away must not decide the framing. */
function sceneBounds(data: LoadedScene) {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const person of data.people) {
    for (let i = 0; i < person.frames.length; i += 12) {
      const [x, y] = person.pelvisAt(i);
      xs.push(x);
      ys.push(y);
    }
  }
  if (xs.length === 0) return new THREE.Box3(new THREE.Vector3(-5, -5, 0), new THREE.Vector3(5, 5, 2));
  const quantile = (values: number[], q: number) => {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.round(q * (sorted.length - 1))];
  };
  return new THREE.Box3(
    new THREE.Vector3(quantile(xs, 0.03), quantile(ys, 0.03), 0),
    new THREE.Vector3(quantile(xs, 0.97), quantile(ys, 0.97), 2),
  );
}

function People({ clock, hidden, selected, solo, display, labels, onSelect, objects, cameraMode, video }: Props & { objects: PersonObject[] }) {
  const group = useRef<THREE.Group>(null);
  useEffect(() => {
    const root = group.current;
    if (!root) return;
    for (const o of objects) root.add(o.root);
    return () => {
      for (const o of objects) root.remove(o.root);
    };
  }, [objects]);

  useFrame(() => {
    const frame = clock.frame();
    for (const o of objects) {
      const id = o.track.entry.id;
      if (hidden.has(id)) {
        o.root.visible = false;
        continue;
      }
      o.update(frame, {
        mesh: display.mesh,
        skeleton: display.skeleton,
        trails: display.trails,
        markers: cameraMode === "top",
        selected: selected === id,
        dimmed: solo && selected !== null && selected !== id,
        opacity: cameraMode === "camera" && video ? 0.82 : 1,
      });
    }
  });

  return (
    <group
      ref={group}
      quaternion={WORLD_TO_THREE}
      onClick={(event) => {
        event.stopPropagation();
        const id = event.object.userData.personId;
        if (typeof id === "number") onSelect(id === selected ? null : id);
      }}
    >
      {display.labels &&
        objects.map((o) =>
          hidden.has(o.track.entry.id) ? null : (
            <primitive key={o.track.entry.id} object={o.labelAnchor}>
              <Html center zIndexRange={[20, 0]} style={{ pointerEvents: "none" }}>
                <span
                  className={`person-label${selected === o.track.entry.id ? " on" : ""}`}
                  style={{ borderColor: o.track.entry.color }}
                >
                  <i style={{ background: o.track.entry.color }} />
                  {labels[o.track.entry.id] ?? o.track.entry.label}
                </span>
              </Html>
            </primitive>
          ),
        )}
    </group>
  );
}

function Rig({ data, clock, selected, cameraMode, objects, bounds }: {
  data: LoadedScene;
  clock: PlaybackClock;
  selected: number | null;
  cameraMode: CameraMode;
  objects: PersonObject[];
  bounds: THREE.Box3;
}) {
  const controls = useRef<OrbitControlsImpl>(null);
  const { camera, size } = useThree();
  const centre = useMemo(() => {
    const c = bounds.getCenter(new THREE.Vector3());
    return toThree(c.x, c.y, 0.9);
  }, [bounds]);
  const cam = data.scene.camera;
  const recordedPosition = useMemo(() => toThree(...cam.position), [cam.position]);
  const pelvis = useMemo(() => new THREE.Vector3(), []);
  const target = useMemo(() => new THREE.Vector3(), []);

  useEffect(() => {
    const perspective = camera as THREE.PerspectiveCamera;
    const extent = bounds.getSize(new THREE.Vector3()).length();
    if (cameraMode === "top") {
      perspective.fov = 40;
      perspective.position.set(centre.x, Math.max(10, extent * 0.8), centre.z + 0.01);
      controls.current?.target.copy(centre).setY(0);
    } else if (cameraMode === "orbit" || cameraMode === "follow") {
      perspective.fov = 40;
      const back = recordedPosition.clone().sub(centre).setY(0).normalize();
      if (!Number.isFinite(back.x)) back.set(0, 0, 1);
      const distance = Math.max(8, extent * 0.62);
      perspective.position.copy(centre).addScaledVector(back, distance).setY(Math.max(4, distance * 0.45));
      controls.current?.target.copy(centre);
    }
    perspective.updateProjectionMatrix();
    controls.current?.update();
  }, [cameraMode, camera, centre, bounds, recordedPosition]);

  useFrame((_, delta) => {
    const perspective = camera as THREE.PerspectiveCamera;
    if (cameraMode === "camera") {
      const q = cam.rotation;
      perspective.position.copy(recordedPosition);
      perspective.quaternion.copy(WORLD_TO_THREE).multiply(new THREE.Quaternion(q[0], q[1], q[2], q[3])).multiply(CAMERA_AXES);
      const fov = (2 * Math.atan(data.scene.height / 2 / cam.focal) * 180) / Math.PI;
      if (Math.abs(perspective.fov - fov) > 1e-3 || perspective.aspect !== size.width / size.height) {
        perspective.fov = fov;
        perspective.updateProjectionMatrix();
      }
      return;
    }
    if (cameraMode === "follow" && selected !== null && controls.current) {
      const person = objects.find((o) => o.track.entry.id === selected);
      if (person?.isVisible) {
        person.pelvis(pelvis);
        toThree(pelvis.x, pelvis.y, 0.9, target);
        const k = 1 - Math.exp(-delta * 4);
        const shift = target.clone().sub(controls.current.target).multiplyScalar(k);
        controls.current.target.add(shift);
        perspective.position.add(shift);
        controls.current.update();
      }
    }
  });

  return (
    <OrbitControls
      ref={controls}
      enabled={cameraMode !== "camera"}
      enableDamping
      dampingFactor={0.12}
      maxPolarAngle={cameraMode === "top" ? 0.35 : Math.PI / 2 - 0.03}
      minDistance={1.5}
      maxDistance={250}
      makeDefault
    />
  );
}

/**
 * The source video behind the 3D people, seen through the recording camera:
 * a plane filling that camera's field of view, so any misalignment shows.
 */
function VideoBackdrop({ video, data }: { video: HTMLVideoElement; data: LoadedScene }) {
  const { camera } = useThree();
  const mesh = useRef<THREE.Mesh>(null);
  const texture = useMemo(() => {
    const t = new THREE.VideoTexture(video);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, [video]);
  useEffect(() => {
    // Video textures refresh on presented frames only; a paused or seeking
    // video presents none, so refresh explicitly.
    const refresh = () => {
      if (video.readyState >= 2) texture.needsUpdate = true;
    };
    refresh();
    const events = ["loadeddata", "seeked", "timeupdate"] as const;
    events.forEach((e) => video.addEventListener(e, refresh));
    return () => {
      events.forEach((e) => video.removeEventListener(e, refresh));
      texture.dispose();
    };
  }, [texture, video]);
  const depth = 200;
  const height = (depth * data.scene.height) / data.scene.camera.focal;
  const width = (depth * data.scene.width) / data.scene.camera.focal;
  useFrame(() => {
    const m = mesh.current;
    if (!m) return;
    m.position.copy(camera.position);
    m.quaternion.copy(camera.quaternion);
    m.translateZ(-depth);
  });
  return (
    <mesh ref={mesh} renderOrder={-1} frustumCulled={false}>
      <planeGeometry args={[width, height]} />
      <meshBasicMaterial map={texture} depthWrite={false} depthTest={false} toneMapped={false} fog={false} />
    </mesh>
  );
}

function Lights({ bounds }: { bounds: THREE.Box3 }) {
  const light = useRef<THREE.DirectionalLight>(null);
  const centre = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const half = Math.max(size.x, size.y) / 2 + 6;
  useEffect(() => {
    const l = light.current;
    if (!l) return;
    l.target.position.copy(toThree(centre.x, centre.y, 0));
    l.target.updateMatrixWorld();
    const cam = l.shadow.camera;
    cam.left = -half;
    cam.right = half;
    cam.top = half;
    cam.bottom = -half;
    cam.near = 1;
    cam.far = 120;
    cam.updateProjectionMatrix();
  }, [centre, half]);
  const sun = toThree(centre.x - 12, centre.y - 18, 32);
  return (
    <>
      <hemisphereLight args={["#ffffff", "#c9ced6", 1.15]} />
      <directionalLight
        ref={light}
        position={sun}
        intensity={1.6}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-bias={-0.0004}
        shadow-normalBias={0.02}
      />
    </>
  );
}

function Floor({ bounds }: { bounds: THREE.Box3 }) {
  const centre = bounds.getCenter(new THREE.Vector3());
  const position = toThree(centre.x, centre.y, 0);
  return (
    <group position={[position.x, 0, position.z]}>
      <mesh rotation-x={-Math.PI / 2} receiveShadow position={[0, -0.001, 0]}>
        <planeGeometry args={[400, 400]} />
        <shadowMaterial opacity={0.14} />
      </mesh>
      <Grid
        args={[400, 400]}
        cellSize={1}
        cellThickness={0.7}
        cellColor="#cdd3dc"
        sectionSize={5}
        sectionThickness={1.2}
        sectionColor="#a9b2bf"
        fadeDistance={90}
        fadeStrength={1.4}
        infiniteGrid
      />
    </group>
  );
}

export function Stage(props: Props) {
  const { data } = props;
  const shared = useMemo(() => new SharedBody(data.mesh), [data.mesh]);
  const objects = useMemo(
    () => data.people.map((track) => new PersonObject(track, shared, data.scene.bones, data.scene.fps, data.scene.frames)),
    [data.people, data.scene, shared],
  );
  useEffect(() => () => objects.forEach((o) => o.dispose()), [objects]);
  const bounds = useMemo(() => sceneBounds(data), [data]);

  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      gl={{ antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: true }}
      camera={{ fov: 40, near: 0.1, far: 600, position: [0, 8, 16] }}
      onPointerMissed={() => props.onSelect(null)}
    >
      <color attach="background" args={["#eef1f5"]} />
      <fog attach="fog" args={["#eef1f5", 60, 260]} />
      <Lights bounds={bounds} />
      <Floor bounds={bounds} />
      <People {...props} objects={objects} />
      {props.cameraMode === "camera" && props.video && <VideoBackdrop video={props.video} data={data} />}
      <Rig data={data} clock={props.clock} selected={props.selected} cameraMode={props.cameraMode} objects={objects} bounds={bounds} />
    </Canvas>
  );
}
