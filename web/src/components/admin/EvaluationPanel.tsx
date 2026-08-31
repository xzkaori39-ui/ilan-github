"use client";

import { useCallback, useEffect, useState } from "react";
import {
  getEvaluationJob, latestEvaluation, runEvaluation,
  type EvaluationMetrics, type EvaluationProfile, type EvaluationJob, type GraphSensitiveMetrics, type RAGEvaluation, type User,
} from "@/lib/api";
import Icon from "../Icon";
import styles from "./admin.module.css";

const QUALITY_ROWS: Array<[keyof EvaluationMetrics, string]> = [
  ["recall_at_k", "Recall@5"], ["mrr", "MRR"], ["ndcg_at_k", "二元 nDCG@5"],
  ["citation_correctness", "引用正确率"], ["answer_key_coverage", "答案关键项覆盖率"],
];
const ENGINEERING_ROWS: Array<[keyof EvaluationMetrics, string, "ms" | "%" | "count" | "yuan"]> = [
  ["avg_latency_ms", "平均时延", "ms"], ["p50_latency_ms", "P50 时延", "ms"], ["p95_latency_ms", "P95 时延", "ms"],
  ["failure_rate", "失败率", "%"], ["graph_usage_rate", "图增强使用率", "%"], ["avg_graph_evidence", "平均图补充证据", "count"],
  ["total_tokens", "Token", "count"], ["cost_yuan", "单问成本", "yuan"],
];

function percent(value: number) { return `${(value * 100).toFixed(2)}%`; }
function metric(value: number | null | undefined, unit: "quality" | "ms" | "%" | "count" | "yuan") {
  if (value === null || value === undefined) return "上游未提供";
  if (unit === "quality" || unit === "%") return percent(value);
  if (unit === "ms") return `${value} ms`;
  if (unit === "yuan") return `¥${value.toFixed(4)}`;
  return value.toLocaleString();
}
function profileByName(evaluation: RAGEvaluation, name: EvaluationProfile["name"]) {
  return evaluation.profiles?.find((profile) => profile.name === name);
}
function configValue(config: Record<string, unknown>, key: string) {
  const value = config[key];
  return typeof value === "string" || typeof value === "number" ? String(value) : "未提供";
}
function PromptHashes({ config }: { config: Record<string, unknown> }) {
  const hashes = config.prompt_hashes;
  if (!hashes || typeof hashes !== "object") return <>未提供</>;
  return <>{Object.entries(hashes as Record<string, unknown>).map(([name, value]) => `${name}:${String(value).slice(0, 12)}`).join(" · ")}</>;
}

export default function EvaluationPanel({ user }: { user: User }) {
  const [evaluation, setEvaluation] = useState<RAGEvaluation | null>(null);
  const [job, setJob] = useState<EvaluationJob | null>(null);
  const [error, setError] = useState("");
  const datasetId = "real_document_qa" as const;
  const isSuper = !user.dept_id;
  const load = useCallback(async () => {
    try { setEvaluation(await latestEvaluation()); setError(""); }
    catch (reason) { setError(String(reason instanceof Error ? reason.message : reason)); }
  }, []);
  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setTimeout(async () => {
      try {
        const next = await getEvaluationJob(job._id);
        setJob(next);
        if (next.status === "completed") await load();
        if (next.status === "failed") setError(next.result?.error || "评测作业失败");
      } catch (reason) { setError(String(reason instanceof Error ? reason.message : reason)); }
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [job, load]);
  const run = async () => {
    try {
      setError("");
      const queued = await runEvaluation(datasetId);
      setJob({ _id: queued.job_id, type: "evaluation_run", status: "queued" });
    } catch (reason) { setError(String(reason instanceof Error ? reason.message : reason)); }
  };
  const baseline = evaluation && profileByName(evaluation, "baseline_no_graph");
  const candidate = evaluation && profileByName(evaluation, "candidate_graph_enabled");

  return <div className={styles.panelStack}>
    <section className={styles.heroPanel}>
      <div><span className={styles.eyebrow}>OFFLINE RAG EVALUATION</span><h2>离线回放，先量化再灰度</h2><p>在公开示例问答集上比较图增强关闭与开启；模型、Embedding、Reranker、Prompt 与 Top-K 保持一致，不混入用户真实会话。</p></div>
      <div className={styles.heroBadge}><Icon name="review" size={28}/><span><b>{evaluation?.case_count ?? "—"}</b><small>示例评测题</small></span></div>
    </section>
    {error && <div className={styles.errorBanner}><Icon name="shield" size={17}/>{error}</div>}
    <section className={styles.card}>
      <div className={styles.rowBetween}><div><h3>评测运行</h3><p className={styles.muted}>两档案仅图增强开关不同；金标准不会发送给模型。</p></div>
        {isSuper ? <button className={styles.btn} disabled={!!job && ["queued", "running"].includes(job.status)} onClick={() => void run()}>{job && ["queued", "running"].includes(job.status) ? "离线回放运行中" : "运行离线回放"}</button> : <span className={`${styles.badge} ${styles.badgeBlue}`}>部门只读视图</span>}
      </div>
      {isSuper && <p className={styles.qaMeta}>评测数据集　公开示例问答集（6题）。自建知识库请按 <code>backend/evaluation/README.md</code> 建立独立金标，不要复用示例结果。</p>}
      {job && <div className={styles.qablock}><b className={styles.qaQuestion}>作业状态：{job.status}</b><div className={styles.qaMeta}>{job.progress?.stage || "queued"} · {job.progress?.detail || "等待 Worker 消费"}</div></div>}
      <p className={styles.cardHint}>Faithfulness 需 Judge，首期未启用；“内置校验”不会被包装为 Faithfulness。Token 与成本仅在模型上游返回 usage 时显示。</p>
    </section>
    {!evaluation ? <div className={styles.empty}>尚无评测记录。系统管理员可发起一次离线回放。</div> : <>
      {baseline && candidate && <section className={styles.card}><div className={styles.cardTitle}><span><Icon name="activity" size={17}/>策略对比</span><em>{evaluation.status.toUpperCase()}</em></div><table className={styles.table}><thead><tr><th>指标</th><th>{baseline.label}</th><th>{candidate.label}</th><th>候选增量</th></tr></thead><tbody>
        {QUALITY_ROWS.map(([key, label]) => <tr key={key}><td>{label}</td><td>{metric(baseline.metrics[key], "quality")}</td><td>{metric(candidate.metrics[key], "quality")}</td><td className={(evaluation.comparison?.deltas[key] || 0) >= 0 ? styles.deltaUp : styles.deltaDown}>{metric(evaluation.comparison?.deltas[key] ?? null, "quality")}</td></tr>)}
        {ENGINEERING_ROWS.map(([key, label, unit]) => <tr key={key}><td>{label}</td><td>{metric(baseline.metrics[key], unit)}</td><td>{metric(candidate.metrics[key], unit)}</td><td>—</td></tr>)}
      </tbody></table><p className={styles.cardHint}>离线建议：{evaluation.comparison?.recommendation === "consider_candidate" ? "候选可进入下一步人工复核" : "保留当前基线"}。公开示例仅用于验证链路，不能代表真实校园制度库效果。</p><div className={styles.grid2}><ConfigSnapshot profile={baseline}/><ConfigSnapshot profile={candidate}/></div></section>}
      {(evaluation.profiles || []).map((profile) => <section className={styles.card} key={profile.name}><div className={styles.cardTitle}><span><Icon name="layers" size={17}/>{profile.label}</span><em>{profile.failed_cases || 0} FAILED</em></div><table className={styles.table}><thead><tr><th>题目</th><th>检索命中</th><th>图补充证据</th><th>引用正确率</th><th>关键项覆盖率</th><th>时延</th><th>状态</th></tr></thead><tbody>{profile.details.map((detail) => <tr key={detail.id}><td><b>{detail.id}</b><br/><span className={styles.muted}>{detail.query}</span></td><td>{detail.hit_at_k ? `第 ${detail.rank} 位` : "未命中"}</td><td>{detail.graph_evidence_count ?? 0}</td><td>{percent(detail.citation_correctness)}</td><td>{percent(detail.answer_key_coverage)}</td><td>{detail.latency_ms} ms</td><td><span className={`${styles.badge} ${detail.success ? styles.badgeGreen : styles.badgeRed}`}>{detail.success ? "完成" : "失败"}</span>{detail.error && <div className={styles.error}>{detail.error}</div>}</td></tr>)}</tbody></table></section>)}
    </>}
  </div>;
}

function answerEvidenceLabel(profile: EvaluationProfile) {
  const topK = configValue(profile.config || {}, "hybrid_topk");
  const graphLimit = configValue(profile.config || {}, "graph_expansion_limit");
  return `回答证据完整率（Top-${topK}${Number(graphLimit) > 0 ? ` + 最多 ${graphLimit} 条图证据` : ""}）`;
}

function GraphMetrics({ title, baseline, candidate, answerEvidenceLabel, rescue }: { title: string; baseline?: GraphSensitiveMetrics; candidate?: GraphSensitiveMetrics; answerEvidenceLabel: string; rescue?: RAGEvaluation["comparison"] extends infer _T ? { graph_rescue_numerator: number; graph_rescue_denominator: number; graph_rescue_rate: number | null } | null : never }) {
  if (!baseline || !candidate) return null;
  const rows: Array<[keyof GraphSensitiveMetrics, string]> = [["evidence_set_complete_in_answer_context", answerEvidenceLabel], ["bridge_evidence_hit_rate", "目标桥接证据命中率"], ["graph_evidence_precision", "图新增证据精确率"], ["graph_path_validity_rate", "图路径有效率"], ["distractor_resistance_rate", "干扰抵抗率"], ["graph_fallback_rate", "图降级率"]];
  return <div className={styles.qablock}><b className={styles.qaQuestion}>{title} · 可解释指标（n={baseline.case_count}）</b><table className={styles.table}><tbody>{rows.map(([key, label]) => <tr key={key}><td>{label}</td><td>{metric(baseline[key] as number | null, "%")}</td><td>{metric(candidate[key] as number | null, "%")}</td></tr>)}{rescue && <tr><td>Graph Rescue Rate</td><td colSpan={2}>{rescue.graph_rescue_rate === null ? "无适用样本" : `${percent(rescue.graph_rescue_rate)}（${rescue.graph_rescue_numerator}/${rescue.graph_rescue_denominator}）`}</td></tr>}</tbody></table></div>;
}

function ConfigSnapshot({ profile }: { profile: EvaluationProfile }) {
  const config = profile.config || {};
  return <div className={styles.qablock}><b className={styles.qaQuestion}>配置快照 · {profile.label}</b><div className={styles.qaMeta}>Chat：{configValue(config, "model")} · TopK：{configValue(config, "hybrid_topk")}</div><div className={styles.qaMeta}>Embedding：{configValue(config, "embedding_provider")} / {configValue(config, "embedding_model")}</div><div className={styles.qaMeta}>Reranker：{config.reranker_enabled ? `${configValue(config, "reranker_provider")} / ${configValue(config, "reranker_model")}` : "关闭"}</div><div className={styles.qaMeta}>图增强：{config.graph_enabled ? `启用（最多 ${configValue(config, "graph_expansion_limit")} 条补充证据）` : "关闭"}</div><div className={styles.qaMeta}>Prompt SHA-256：<PromptHashes config={config}/></div></div>;
}
