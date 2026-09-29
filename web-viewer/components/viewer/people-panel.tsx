"use client";

import { useState } from "react";

import type { PersonEntry } from "@/lib/scene/types";
import { Download, Eye, EyeOff, Target } from "../icons";

type Props = {
  people: PersonEntry[];
  labels: Record<number, string>;
  hidden: Set<number>;
  selected: number | null;
  solo: boolean;
  visibleNow: Set<number>;
  onSelect: (id: number | null) => void;
  onToggleHidden: (id: number) => void;
  onShowOnly: (ids: number[] | null) => void;
  onSolo: (value: boolean) => void;
  onRename: (id: number, label: string) => void;
  onExport: () => void;
};

export function PeoplePanel(props: Props) {
  const { people, labels, hidden, selected, visibleNow } = props;
  const [editing, setEditing] = useState<number | null>(null);
  const [briefHidden, setBriefHidden] = useState(false);
  const ordered = [...people].sort((a, b) => a.first - b.first);
  const brief = people.filter((p) => p.summary.visible_seconds < 1).map((p) => p.id);

  return (
    <section className="panel people-panel">
      <header className="panel-head">
        <h3>
          People <span className="faint">· {people.length}</span>
        </h3>
        <span className="faint tabular" title="On screen right now">
          {visibleNow.size} in view
        </span>
      </header>
      <div className="panel-tools">
        <button
          className={`btn btn-sm${props.solo ? " btn-active" : ""}`}
          disabled={selected === null}
          onClick={() => props.onSolo(!props.solo)}
          title="Fade everyone except the selected person"
        >
          <Target size={14} /> Focus
        </button>
        {brief.length > 0 && (
          <button
            className="btn btn-sm"
            onClick={() => {
              props.onShowOnly(briefHidden ? null : people.map((p) => p.id).filter((id) => !brief.includes(id)));
              setBriefHidden(!briefHidden);
            }}
            title="People seen for less than a second are usually fragments or passers-by"
          >
            {briefHidden ? "Show" : "Hide"} {brief.length} brief
          </button>
        )}
        {hidden.size > 0 && (
          <button className="btn btn-sm btn-ghost" onClick={() => props.onShowOnly(null)}>
            Show all
          </button>
        )}
        <button className="btn btn-sm btn-ghost push-right" onClick={props.onExport} title="Every person's measures as JSON">
          <Download size={14} /> Export
        </button>
      </div>
      <ul className="people-list">
        {ordered.map((p) => {
          const label = labels[p.id] ?? p.label;
          const isHidden = hidden.has(p.id);
          return (
            <li
              key={p.id}
              className={`person-row${selected === p.id ? " on" : ""}${isHidden ? " off" : ""}`}
              onClick={() => props.onSelect(selected === p.id ? null : p.id)}
            >
              <span className={`swatch${visibleNow.has(p.id) ? " live" : ""}`} style={{ background: p.color }} />
              {editing === p.id ? (
                <input
                  className="input rename"
                  autoFocus
                  defaultValue={label}
                  onClick={(e) => e.stopPropagation()}
                  onBlur={(e) => {
                    props.onRename(p.id, e.target.value);
                    setEditing(null);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                    if (e.key === "Escape") setEditing(null);
                  }}
                />
              ) : (
                <span
                  className="person-name"
                  onDoubleClick={(e) => {
                    e.stopPropagation();
                    setEditing(p.id);
                  }}
                  title="Double-click to rename"
                >
                  {label}
                </span>
              )}
              <span className="person-meta faint tabular" title="Time on screen · distance covered">
                {p.summary.visible_seconds.toFixed(0)} s · {p.summary.distance_m.toFixed(0)} m
              </span>
              <button
                className="icon-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  props.onToggleHidden(p.id);
                }}
                title={isHidden ? "Show" : "Hide"}
              >
                {isHidden ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
