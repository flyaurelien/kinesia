/** Shape of `scene/scene.json`, written by `kinesia.scene.build`. */

export type SectionType = "uint8" | "uint16" | "int16" | "int32" | "float32";

export type Section = {
  offset: number;
  type: SectionType;
  shape: number[];
};

export type PersonLayout = {
  rest_vertices: Section;
  joint_scales: Section;
  frames: Section;
  pelvis: Section;
  local_q: Section;
  local_t: Section;
  keypoints: Section;
  flags: Section;
};

export type Jump = { frame: number; height: number; duration: number };

export type PersonSummary = {
  visible_seconds: number;
  distance_m: number;
  top_speed: number;
  mean_speed: number;
  height_m: number;
  jumps: Jump[];
  highest_jump_m: number;
};

export type PersonEntry = {
  id: number;
  label: string;
  color: string;
  tracks: number[];
  segments: [number, number][];
  first: number;
  last: number;
  measured: number;
  summary: PersonSummary;
  layout: PersonLayout;
};

export type SceneFile = {
  version: number;
  fps: number;
  frames: number;
  width: number;
  height: number;
  camera: {
    focal: number;
    focal_source: string;
    /** Fixed camera: position (world metres) and camera-to-world rotation (x, y, z, w). */
    position: [number, number, number];
    rotation: [number, number, number, number];
  };
  floor: { camera_height: number };
  mesh: {
    file: string;
    vertices: number;
    layout: Record<"faces" | "skin_index" | "skin_weight" | "inverse_bind" | "parents", Section>;
  };
  people_file: string;
  encoding: {
    quat_scale: number;
    local_t_unit: number;
    keypoint_unit: number;
    flags: { measured: number; left_contact: number; right_contact: number };
    pelvis_joint: number;
  };
  bones: [number, number][];
  people: PersonEntry[];
  identity: { threshold: number; fragments: number; links: { from: number; to: number; similarity: number; gap_frames: number }[] };
};

export type OverlayFile = {
  /** Per frame: `[person, x0, y0, x1, y1, polygons]` entries. */
  frames: [number, number, number, number, number, number[][]][][];
};

export type MetricsFile = {
  series_labels: Record<string, string>;
  people: { summary: PersonSummary; series: Record<string, number[]> }[];
};

/** `edits.json`: user corrections, applied on top of the built scene. */
export type Edits = {
  labels?: Record<string, string>;
  hidden?: number[];
};
