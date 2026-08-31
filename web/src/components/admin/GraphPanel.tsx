"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { graphView, type GraphEdge, type GraphNode } from "@/lib/api";
import Icon from "../Icon";
import styles from "./admin.module.css";

const WIDTH = 780, HEIGHT = 450;
const COLORS: Record<string, string> = {
  Entity: "#2563eb", Department: "#0f766e", Document: "#b7791f", Chunk: "#64748b",
  TextUnit: "#7c3aed", Community: "#db2777", CommunityReport: "#c2410c", Topic: "#0891b2",
};
type Point = { x: number; y: number };
type Drag = { nodeId?: string; x: number; y: number } | null;

function clamp(value: number, low: number, high: number) { return Math.max(low, Math.min(high, value)); }
function stableNumber(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index++) hash = Math.imul(hash ^ value.charCodeAt(index), 16777619);
  return (hash >>> 0) / 4294967295;
}

/** Deterministic force layout: no dependency or random redraw on refresh. */
function forceLayout(nodes: GraphNode[], edges: GraphEdge[], focusId?: string): Record<string, Point> {
  const center = { x: WIDTH / 2, y: HEIGHT / 2 };
  const points: Record<string, Point> = {};
  nodes.forEach((node, index) => {
    const angle = stableNumber(`${node.id}:a`) * Math.PI * 2;
    const radius = 55 + stableNumber(`${node.id}:r`) * 150;
    points[node.id] = node.id === focusId ? { ...center } : {
      x: center.x + Math.cos(angle) * radius + (index % 3) * 8,
      y: center.y + Math.sin(angle) * radius,
    };
  });
  const validEdges = edges.filter(edge => points[edge.source] && points[edge.target]);
  for (let iteration = 0; iteration < 130; iteration++) {
    const movement: Record<string, Point> = Object.fromEntries(nodes.map(node => [node.id, { x: 0, y: 0 }]));
    for (let left = 0; left < nodes.length; left++) for (let right = left + 1; right < nodes.length; right++) {
      const a = points[nodes[left].id], b = points[nodes[right].id];
      const dx = a.x - b.x, dy = a.y - b.y, distance2 = Math.max(dx * dx + dy * dy, 40);
      const push = 620 / distance2;
      movement[nodes[left].id].x += dx * push; movement[nodes[left].id].y += dy * push;
      movement[nodes[right].id].x -= dx * push; movement[nodes[right].id].y -= dy * push;
    }
    for (const edge of validEdges) {
      const a = points[edge.source], b = points[edge.target];
      const dx = b.x - a.x, dy = b.y - a.y, distance = Math.max(Math.hypot(dx, dy), 1);
      const pull = (distance - 92) * 0.009;
      movement[edge.source].x += dx / distance * pull; movement[edge.source].y += dy / distance * pull;
      movement[edge.target].x -= dx / distance * pull; movement[edge.target].y -= dy / distance * pull;
    }
    for (const node of nodes) {
      const point = points[node.id], toCenter = node.id === focusId ? 0.035 : 0.006;
      movement[node.id].x += (center.x - point.x) * toCenter; movement[node.id].y += (center.y - point.y) * toCenter;
      point.x = clamp(point.x + movement[node.id].x, 34, WIDTH - 34);
      point.y = clamp(point.y + movement[node.id].y, 34, HEIGHT - 34);
    }
  }
  return points;
}

export default function GraphPanel() {
  const [query, setQuery] = useState("");
  const [label, setLabel] = useState("");
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [summary, setSummary] = useState<{ nodes: Record<string, number>; edges: number } | null>(null);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [positions, setPositions] = useState<Record<string, Point>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const drag = useRef<Drag>(null);

  const load = useCallback(async (nextQuery = "", nextLabel = "") => {
    setLoading(true);
    try {
      const result = await graphView(nextQuery, nextLabel);
      setNodes(result.nodes); setEdges(result.edges);
      setSummary({ nodes: result.summary?.nodes || {}, edges: result.summary?.edges || 0 });
      setSelected(nextQuery.trim() ? result.nodes[0] || null : null);
      setZoom(1); setPan({ x: 0, y: 0 });
      setError(result.connected ? "" : "图增强不可用，当前页面无法读取 Neo4j 图谱");
    } catch (event) {
      setError(event instanceof Error ? event.message : String(event)); setNodes([]); setEdges([]);
    } finally { setLoading(false); }
  }, []);
  useEffect(() => { void load(); }, [load]);

  const focusId = selected?.id || (query.trim() ? nodes[0]?.id : undefined);
  const layout = useMemo(() => forceLayout(nodes, edges, focusId), [nodes, edges, focusId]);
  useEffect(() => { setPositions(layout); }, [layout]);
  const selectedEdges = selected ? edges.filter(edge => edge.source === selected.id || edge.target === selected.id) : [];
  const labels = useMemo(() => Object.keys(summary?.nodes || {}).sort(), [summary]);
  const activeIds = useMemo(() => new Set(selectedEdges.flatMap(edge => [edge.source, edge.target])), [selectedEdges]);
  const resetView = () => { setZoom(1); setPan({ x: 0, y: 0 }); setPositions(layout); };

  return <div className={styles.panelStack}>
    <section className={styles.heroPanel}>
      <div><span className={styles.eyebrow}>GRAPH OBSERVABILITY</span><h2>知识图谱可视化</h2><p>搜索一个实体后，以它为中心展开一跳关系；图谱只读，图增强不可用时不影响普通 RAG 问答。</p></div>
      <div className={styles.heroBadge}><Icon name="layers" size={24}/><div><b>{summary ? Object.values(summary.nodes).reduce((a, b) => a + b, 0) : "—"}</b><small>图谱节点</small></div></div>
    </section>
    {error && <div className={styles.errorBanner}><Icon name="shield" size={17}/>{error}</div>}
    <section className={styles.card}>
      <div className={styles.sectionHead}><div><span className={styles.eyebrow}>READ ONLY SUBGRAPH</span><h2>局部关系网络</h2><p>总览限制为 45 节点 / 90 条边；搜索后自动聚焦结果。滚轮缩放，拖拽空白处平移，拖动节点可整理布局。</p></div><button className={`${styles.btnGhost} ${styles.btnSm}`} onClick={() => void load(query, label)} disabled={loading}><Icon name="refresh" size={14}/>刷新</button></div>
      <form className={styles.graphToolbar} onSubmit={event => { event.preventDefault(); void load(query, label); }}>
        <div className={styles.graphSearch}><Icon name="search" size={14}/><input className={styles.input} value={query} onChange={event => setQuery(event.target.value)} placeholder="搜索实体、文档或部门，例如：学生申诉" /></div>
        <select className={styles.select} value={label} onChange={event => { setLabel(event.target.value); void load(query, event.target.value); }} aria-label="节点类型"><option value="">全部节点类型</option>{labels.map(item => <option key={item} value={item}>{item}</option>)}</select>
        <button className={styles.btn} type="submit" disabled={loading}>聚焦查询</button>
      </form>
      <div className={styles.graphStats}><span>当前节点 <b>{nodes.length}</b></span><span>当前关系 <b>{edges.length}</b></span><span>全图关系 <b>{summary?.edges ?? "—"}</b></span>{labels.slice(0, 6).map(item => <span key={item}><i style={{ background: COLORS[item] || "#64748b" }}/>{item} {summary?.nodes[item] ?? 0}</span>)}</div>
      <div className={styles.graphLayout}>
        <div className={styles.graphCanvas}>
          {loading ? <div className={styles.loading}><span/><b>正在读取图谱</b><small>Neo4j 只读查询中</small></div> : nodes.length === 0 ? <div className={styles.empty}>没有找到匹配的图谱节点</div> : <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="可缩放知识图谱关系网络" onWheel={event => { event.preventDefault(); setZoom(value => clamp(value * (event.deltaY < 0 ? 1.12 : 0.89), 0.55, 2.6)); }} onPointerMove={event => { const current = drag.current; if (!current) return; const dx = event.clientX - current.x, dy = event.clientY - current.y; current.x = event.clientX; current.y = event.clientY; if (current.nodeId) setPositions(value => ({ ...value, [current.nodeId!]: { x: clamp((value[current.nodeId!]?.x || 0) + dx / zoom, 26, WIDTH - 26), y: clamp((value[current.nodeId!]?.y || 0) + dy / zoom, 26, HEIGHT - 26) } })); else setPan(value => ({ x: value.x + dx, y: value.y + dy })); }} onPointerUp={() => { drag.current = null; }} onPointerLeave={() => { drag.current = null; }}>
            <defs><marker id="ilanGraphArrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#9aa9bf"/></marker></defs>
            <rect className={styles.graphBackground} x="0" y="0" width={WIDTH} height={HEIGHT} onPointerDown={event => { drag.current = { x: event.clientX, y: event.clientY }; }}/>
            <g transform={`translate(${pan.x} ${pan.y}) scale(${zoom})`}>
              {edges.map((edge, index) => { const source = positions[edge.source], target = positions[edge.target]; if (!source || !target) return null; const active = selected && (edge.source === selected.id || edge.target === selected.id); return <g key={`${edge.key || edge.type}-${index}`}><line className={active ? styles.graphEdgeActive : styles.graphEdge} x1={source.x} y1={source.y} x2={target.x} y2={target.y} markerEnd="url(#ilanGraphArrow)"/>{active && <text className={styles.graphEdgeLabel} x={(source.x + target.x) / 2} y={(source.y + target.y) / 2}>{edge.type}</text>}</g>; })}
              {nodes.map(node => { const point = positions[node.id]; if (!point) return null; const active = selected?.id === node.id, neighbor = activeIds.has(node.id), showLabel = active || hovered === node.id || neighbor; return <g key={node.id} className={styles.graphNode} onPointerDown={event => { event.stopPropagation(); setSelected(node); drag.current = { nodeId: node.id, x: event.clientX, y: event.clientY }; }} onPointerEnter={() => setHovered(node.id)} onPointerLeave={() => setHovered(null)} tabIndex={0} role="button" aria-label={`查看 ${node.name}`}><circle cx={point.x} cy={point.y} r={active ? 20 : neighbor ? 16 : 12} fill={COLORS[node.label] || "#64748b"} stroke={active ? "#d8a84e" : "#fff"} strokeWidth={active ? 4 : 2}/><text x={point.x} y={point.y + 3} textAnchor="middle" fill="#fff" fontSize={active ? "8" : "6"}>{node.label.slice(0, 3)}</text>{showLabel && <text className={styles.graphNodeLabel} x={point.x} y={point.y + 29} textAnchor="middle">{node.name.length > 15 ? `${node.name.slice(0, 15)}…` : node.name}</text>}</g>; })}
            </g>
          </svg>}
          {!loading && nodes.length > 0 && <div className={styles.graphControls}><button onClick={() => setZoom(value => clamp(value * 1.18, 0.55, 2.6))}>＋</button><button onClick={() => setZoom(value => clamp(value / 1.18, 0.55, 2.6))}>－</button><button onClick={resetView}>重置视图</button></div>}
        </div>
        <aside className={styles.graphDetail}>{selected ? <><span className={styles.eyebrow}>SELECTED NODE</span><h3>{selected.name}</h3><span className={`${styles.badge} ${styles.badgeBlue}`}>{selected.label}</span><div className={styles.graphPropertyList}>{Object.entries(selected.properties).slice(0, 8).map(([key, value]) => <span key={key}><b>{key}</b><small>{String(value)}</small></span>)}</div><h4>一跳关系（{selectedEdges.length}）</h4>{selectedEdges.slice(0, 12).map((edge, index) => <div className={styles.graphRelation} key={`${edge.key || edge.type}-${index}`}><b>{edge.type}</b><small>{edge.source === selected.id ? "→" : "←"} {nodes.find(node => node.id === (edge.source === selected.id ? edge.target : edge.source))?.name || "关联节点"}</small>{typeof edge.properties.description === "string" && <em>{edge.properties.description}</em>}</div>)}</> : <div className={styles.empty}><Icon name="search" size={22}/><p>搜索后点击节点，查看一跳关系与关系描述</p></div>}</aside>
      </div>
    </section>
  </div>;
}
