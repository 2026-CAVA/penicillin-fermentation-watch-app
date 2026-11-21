import { useEffect, useState } from 'react'
import {
  Area, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

export const get = (path, body) =>
  fetch('/api' + path, body && {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }).then((r) => (r.ok ? r.json() : r.text().then((x) => Promise.reject(new Error(`${r.status} ${path} ${x}`)))))

// GET (or POST when body is given) path; re-fetches when path/body change. null path = skip.
export function useApi(path, body) {
  const [s, set] = useState({ loading: true })
  const key = JSON.stringify(body)
  useEffect(() => {
    if (!path) return
    let live = true
    set({ loading: true })
    get(path, body).then((data) => live && set({ data }), (error) => live && set({ error }))
    return () => { live = false }
  }, [path, key]) // eslint-disable-line react-hooks/exhaustive-deps
  return s
}

export const withT = (path, t, more = '') => {
  const q = [t == null ? '' : `t=${t}`, more].filter(Boolean).join('&')
  return q ? `${path}?${q}` : path
}

export const fmt = (v, d = 1) =>
  v == null || Number.isNaN(v) ? '–' : v.toLocaleString('ko-KR', { minimumFractionDigits: d, maximumFractionDigits: d })

export const STATE = { normal: '정상', caution: '주의', anomaly: '이상' }
export const GRADE_STATE = { STABLE: 'normal', WARNING: 'caution', CRITICAL: 'anomaly' }
export const COLOR = { normal: 'var(--stable)', caution: 'var(--warning)', anomaly: 'var(--critical)' }
export const gradeOf = (score, [warn, crit]) =>
  score >= crit ? ['CRITICAL', 'anomaly'] : score >= warn ? ['WARNING', 'caution'] : ['NORMAL', 'normal']

// runs of one state: [{start, end, state}] from parallel time / state arrays
export function segmentsOf(ts, states) {
  const out = []
  states.forEach((s, i) => {
    if (out.length && out.at(-1).state === s) out.at(-1).end = ts[i]
    else out.push({ start: ts[i], end: ts[i], state: s })
  })
  return out
}

export function Load({ s, children, height = 160 }) {
  if (s.error) return <div className="card anomaly">API 오류: {s.error.message}</div>
  if (s.loading || !s.data) return <div className="empty" style={{ minHeight: height }}>불러오는 중…</div>
  return children(s.data)
}

export function PageHead({ title, sub, right }) {
  return (
    <div className="title-row">
      <div><h1>{title}</h1><div className="muted small">{sub}</div></div>
      {right}
    </div>
  )
}

export const Pill = ({ state, children }) => <span className={'pill ' + state}>{children ?? STATE[state]}</span>

const GRADE = { normal: 'NORMAL', caution: 'WARNING', anomaly: 'CRITICAL' }

// state: the backend's grade (alarm needs the score held for 1 h), else graded from the score alone
export function Gauge({ score, cuts, state, size = 200 }) {
  const cls = state ?? gradeOf(score ?? 0, cuts)[1], grade = GRADE[cls]
  const arc = 'M20 100 A80 80 0 0 1 180 100'
  return (
    <div className="center">
      <div className="gauge" style={{ width: size }}>
        <svg viewBox="0 0 200 110">
          <path d={arc} fill="none" stroke="var(--muted)" strokeWidth="18" strokeLinecap="round" />
          {score > 0 && <path d={arc} fill="none" stroke={COLOR[cls]} strokeWidth="18" strokeLinecap="round"
            pathLength="100" strokeDasharray={`${score} 100`} />}
        </svg>
        <div className="gauge-value"><b className="num">{Math.round(score ?? 0)}</b><span className="muted small">/ 100</span></div>
      </div>
      <span className={'pill plain ' + cls}>{grade}</span>
    </div>
  )
}

const tick = { fontSize: 11, fill: 'var(--muted-foreground)' }

// One variable over time. rows: [{t, v, band?: [lo, hi]}]; segments shade caution/anomaly;
// safe = [lo, hi] dashed lines; cursor = vertical line (Replay / "now").
export function SeriesChart({ rows, unit, label, segments = [], safe, cursor, domain, height = 260, dots }) {
  const [x0, x1] = domain ?? [rows[0]?.t ?? 0, rows.at(-1)?.t ?? 1]
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 10, right: 8, left: -6, bottom: 0 }}>
        <XAxis dataKey="t" type="number" domain={[x0, x1]} allowDataOverflow tickFormatter={(x) => `${Math.round(x)}h`} tick={tick} stroke="var(--border)" />
        <YAxis domain={['auto', 'auto']} tick={tick} stroke="var(--border)" tickFormatter={(v) => fmt(v, 1)} width={56} />
        {segments.filter((s) => s.state !== 'normal' && s.end > x0 && s.start < x1).map((s) => (
          <ReferenceArea key={s.start} x1={Math.max(s.start, x0)} x2={Math.min(s.end, x1)} ifOverflow="hidden"
            fill={COLOR[s.state]} fillOpacity={0.14} />
        ))}
        {safe?.map((y, i) => y != null && <ReferenceLine key={i} y={y} stroke="var(--critical)" strokeDasharray="4 3" strokeOpacity={0.6} />)}
        <Tooltip
          formatter={(v, n) => [Array.isArray(v) ? v.map((x) => fmt(x, 3)).join(' – ') : `${fmt(v, 3)} ${unit}`, n === 'band' ? '정상 범위' : label]}
          labelFormatter={(x) => `${fmt(x)} h`} contentStyle={{ borderRadius: 12, fontSize: 12 }} />
        <Area dataKey="band" stroke="none" fill="var(--stable)" fillOpacity={0.25} isAnimationActive={false} />
        <Line dataKey="v" stroke="var(--primary)" strokeWidth={2} dot={!!dots} connectNulls isAnimationActive={false} />
        {cursor != null && <ReferenceLine x={cursor} stroke="var(--foreground)" strokeDasharray="2 3" />}
      </ComposedChart>
    </ResponsiveContainer>
  )
}
