import { useEffect, useMemo, useState } from 'react'
import {
  Area, Bar, BarChart, Cell, ComposedChart, Legend, Line, LineChart, ReferenceArea, ReferenceDot, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import {
  COLOR, GRADE_STATE, Gauge, Load, PageHead, Pill, SeriesChart, fmt, get, segmentsOf, useApi, withT,
} from './lib.jsx'

const tick = { fontSize: 11, fill: 'var(--muted-foreground)' }
const tip = { borderRadius: 12, fontSize: 12 }
const More = ({ go, to }) => <button className="link" onClick={() => go(to)}>자세히 →</button>
const SHAP_NOTE = '기여도(SHAP)는 모델 예측에 대한 영향이며, 물리적 인과관계를 증명하지 않습니다.'

// --- Overview ------------------------------------------------------------------------------------

export function Overview({ id, t, fullH, model, go }) {
  const detail = useApi(withT(`/batches/${id}`, t))
  const forecast = useApi((t ?? fullH) > 0 && `/batches/${id}/forecast?t=${t ?? fullH}`)
  const shap = useApi(withT(`/batches/${id}/shap`, t, 'k=3'))
  const events = useApi(withT(`/batches/${id}/events`, t))
  const opt = useApi(`/batches/${id}/optimize`, {})
  return (
    <Load s={detail}>{(d) => {
      const nowState = d.segments.at(-1)?.state ?? d.state
      return (
        <>
          <PageHead title={`Batch #${d.id}`} sub={`Penicillin (IndPenSim) · 경과 ${fmt(d.elapsedH)} h · 측정 간격 ${d.intervalH} h`} />
          <div className="grid">
            <div className="card span-4">
              <h2>위험도 Risk Score <More go={go} to="risk" /></h2>
              <Gauge score={d.latestScore} cuts={d.scoreCuts} state={nowState} />
              <p className="explain">{d.explanation}</p>
              <p className="foot center">주의 ≥{d.scoreCuts[0]} · 위험 ≥{d.scoreCuts[1]} · 최초 이상 {d.firstAnomalyH == null ? '없음' : `${fmt(d.firstAnomalyH)} h`}</p>
            </div>
            <div className="card span-4">
              <h2>8시간 농도 예측 <More go={go} to="prediction" /></h2>
              <Load s={forecast}>{(f) => <ForecastMini d={d} f={f} />}</Load>
            </div>
            <div className="span-4 stack">
              <div className="tiles">
                <div className="tile"><span className="muted">공정 상태</span><Pill state={nowState} /></div>
                <div className="tile"><span className="muted">{d.runStatus === 'running' ? '경과 시간' : '완료 시점'}</span><b className="num">{fmt(d.elapsedH)}h</b></div>
                <div className="tile"><span className="muted">이상 이벤트</span>
                  <b className={'num ' + (events.data?.length ? 'down' : '')}>{events.data ? `${events.data.length}건` : '–'}</b></div>
              </div>
              <div className="card">
                <h2>Root Cause Top 3 <More go={go} to="root-cause" /></h2>
                <Load s={shap}>{(s) => <ShapBars items={s} cutH={model.target.predicted_at_h} />}</Load>
              </div>
            </div>
            <div className="card span-8">
              <h2>공정 모니터링 Process Monitoring <More go={go} to="monitoring" /></h2>
              <RecentProcess d={d} />
            </div>
            <div className="card span-4">
              <h2>최적 운전조건 추천 <More go={go} to="optimization" /></h2>
              <Load s={opt}>{(o) => <OptimizeMini o={o} />}</Load>
            </div>
          </div>
        </>
      )
    }}</Load>
  )
}

function ForecastMini({ d, f }) {
  const last = f.path.at(-1)
  const data = f.path.map((p) => ({ t: p.t, p: p.p, band: [p.lo, p.hi] }))
  const [lo, hi] = d.predictedRange ?? []
  return (
    <>
      <div className="big num">{fmt(last.p, 2)}<small>{f.unit}</small></div>
      <div className="muted small">80% 구간 {fmt(last.lo, 2)} – {fmt(last.hi, 2)} {f.unit} · t+8 h</div>
      <ResponsiveContainer width="100%" height={90}>
        <ComposedChart data={data} margin={{ top: 8, right: 0, left: 0, bottom: 0 }}>
          <XAxis dataKey="t" hide /><YAxis hide domain={['auto', 'auto']} />
          <Tooltip formatter={(v) => (Array.isArray(v) ? v.map((x) => fmt(x, 2)).join(' – ') : fmt(v, 2))} labelFormatter={(x) => `${x} h`} contentStyle={tip} />
          <Area dataKey="band" stroke="none" fill="var(--primary)" fillOpacity={0.12} isAnimationActive={false} />
          <Line dataKey="p" stroke="var(--primary)" strokeWidth={2} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="yield-row">
        <span className="muted">최종 수율 예측</span>
        <b className="num">{fmt(d.predictedFinal, 0)} kg {lo != null && <span className="muted small">({fmt(lo, 0)}–{fmt(hi, 0)})</span>}</b>
      </div>
      {d.finalMeasured != null && <div className="yield-row"><span className="muted">실제 최종 수율</span><b className="num">{fmt(d.finalMeasured, 0)} kg</b></div>}
    </>
  )
}

function ShapBars({ items, cutH }) {
  const max = Math.max(...items.map((s) => Math.abs(s.value)), 1)
  return (
    <>
      {items.length === 0 && <div className="empty">첫 예측 시점 전이라 기여도가 아직 없습니다.</div>}
      {items.map((s, i) => (
        <div className="rc" key={s.feature} title={s.feature}>
          <span className="rank">{i + 1}</span>
          <span className="label">{s.label}</span>
          <b className={'num ' + (s.value > 0 ? 'up' : 'down')}>{s.value > 0 ? '▲ +' : '▼ '}{fmt(s.value, 0)} kg</b>
          <div className="bar"><i style={{ width: `${(Math.abs(s.value) / max) * 100}%`, background: s.value > 0 ? 'var(--stable)' : 'var(--critical)' }} /></div>
        </div>
      ))}
      <p className="foot">최종 수율 예측(kg, {cutH} h 기준)에 대한 기여도입니다. {SHAP_NOTE}</p>
    </>
  )
}

const WINDOW_H = 30

function RecentProcess({ d }) {
  const keys = Object.keys(d.seriesMeta)
  const [k, setK] = useState(keys[0])
  const meta = d.seriesMeta[k]
  const start = Math.max(0, d.elapsedH - WINDOW_H)
  const rows = d.series.filter((p) => p.t >= start).map((p) => ({ t: p.t, v: p[k] }))
  return (
    <>
      <div className="tabs">
        {keys.map((x) => <button key={x} className={x === k ? 'on' : ''} onClick={() => setK(x)}>{d.seriesMeta[x].label}</button>)}
      </div>
      <div className="big num">{fmt(rows.findLast((r) => r.v != null)?.v, 2)}<small>{meta.unit}</small>
        <span className="muted small"> · 최근 {WINDOW_H}시간{meta.kind === 'offline' && `, offline ${meta.delay_h} h 지연`}</span></div>
      <SeriesChart rows={rows} unit={meta.unit} label={meta.label} segments={d.segments} domain={[start, d.elapsedH]} height={240} />
      <p className="foot">음영: 주황 = 주의, 빨강 = 이상 구간</p>
    </>
  )
}

function OptimizeMini({ o }) {
  return (
    <>
      <div className="opt">
        {Object.keys(o.best).map((x) => (
          <div key={x}>
            <span className="muted">{o.labels[x]}</span>
            <b className="num">{fmt(o.current[x], 1)} → {fmt(o.best[x], 1)}</b>
            <span className={'small num ' + (o.change_pct[x] >= 0 ? 'up' : 'down')}> {o.change_pct[x] >= 0 ? '+' : ''}{fmt(o.change_pct[x])}%</span>
          </div>
        ))}
      </div>
      <div className="yield-row">
        <span className="muted">예상 개선</span>
        <b className="num up">+{fmt(o.gain_kg, 0)} kg <span className="muted small">({o.models_agree}/{o.n_models} 모델 동의)</span></b>
      </div>
      <p className="foot">추천값이며 실제 공정에 적용되지 않았습니다. ±{o.max_change * 100}% 이내.</p>
    </>
  )
}

// --- Batch selection -----------------------------------------------------------------------------

const FILTERS = [['all', '전체'], ['running', 'Running'], ['normal', '정상'], ['caution', '주의'], ['anomaly', '이상']]

export function Batches({ id, t, setId, go }) {
  const list = useApi(withT('/batches', t))
  const [q, setQ] = useState('')
  const [f, setF] = useState('all')
  const hit = (b, f) => f === 'all' || (f === 'running' ? b.runStatus === 'running' : b.state === f)
  return (
    <>
      <PageHead title="배치 선택" sub="Batch Selection · 선택한 배치는 모든 화면에 동기화됩니다. 상태는 해당 시점까지의 최악 등급입니다." />
      <input className="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="배치 번호 검색… (예: 92)" />
      <Load s={list}>{(bs) => (
        <>
          <div className="chips">
            {FILTERS.map(([v, l]) => (
              <button key={v} className={f === v ? 'on' : ''} onClick={() => setF(v)}>{l} ({bs.filter((b) => hit(b, v)).length})</button>
            ))}
          </div>
          <div className="batch-grid">
            {bs.filter((b) => hit(b, f) && String(b.id).includes(q.replace('#', '').trim())).map((b) => (
              <button key={b.id} className={'card batch' + (b.id === id ? ' sel' : '')} onClick={() => { setId(b.id); go('overview') }}>
                <div>
                  <div className="row"><span className="dot" style={{ background: COLOR[b.state] }} /><b>#{b.id}</b>
                    <span className={'pill ' + (b.runStatus === 'running' ? 'info' : 'normal')}>{b.runStatus === 'running' ? 'Running' : 'Completed'}</span></div>
                  <p className="muted small">예측 수율 {fmt(b.predictedFinal, 0)} kg{b.finalMeasured != null && ` · 실제 ${fmt(b.finalMeasured, 0)} kg`}</p>
                  <Timeline segments={b.segments} end={b.elapsedH} />
                </div>
                <div className="right">
                  <span className="muted small">최근 위험도</span>
                  <b className="num big">{Math.round(b.latestScore)}</b>
                  <Pill state={b.state} />
                </div>
              </button>
            ))}
          </div>
        </>
      )}</Load>
    </>
  )
}

function Timeline({ segments, end }) {
  return (
    <div className="timeline" title="상태 구간 (0 h → 현재)">
      {segments.map((s) => <i key={s.start} style={{ left: `${(s.start / end) * 100}%`, width: `${((s.end - s.start) / end) * 100}%`, background: COLOR[s.state] }} />)}
    </div>
  )
}

// --- Process monitoring --------------------------------------------------------------------------

const TILE_VARS = ['T', 'pH', 'DO', 'Fs', 'OUR', 'CER']

const bandRows = (v, key, from = -Infinity, to = Infinity) => {
  const s = v.series[key]
  return v.t.map((t, i) => ({ t, v: s.values[i], band: s.band_lo ? [s.band_lo[i], s.band_hi[i]] : undefined }))
    .filter((r) => r.t >= from && r.t <= to)
}
const viewSegments = (v) => segmentsOf(v.t, v.risk.grade.map((g) => GRADE_STATE[g]))
const outOfRange = (s, i) => {
  const x = s.values[i]
  return x != null && ((s.band_flag && s.band_flag[i]) || (s.safe_lo != null && x < s.safe_lo) || (s.safe_hi != null && x > s.safe_hi))
}

function Spark({ values, color }) {
  const vs = values.filter((x) => x != null)
  if (vs.length < 2) return null
  const lo = Math.min(...vs), hi = Math.max(...vs), r = hi - lo || 1
  const pts = vs.map((x, i) => `${(i / (vs.length - 1)) * 100},${24 - ((x - lo) / r) * 22 - 1}`).join(' ')
  return <svg viewBox="0 0 100 24" preserveAspectRatio="none" className="spark"><polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" /></svg>
}

export function Monitoring({ id, t }) {
  const view = useApi(withT(`/batches/${id}/view`, t))
  const [key, setKey] = useState('T')
  return (
    <>
      <PageHead title="공정 모니터링" sub="Process Monitoring · 녹색 영역은 정상 배치 기준 ±3σ 밴드, 점선은 안전 범위, 음영은 이상 구간" />
      <Load s={view} height={300}>{(v) => {
        const n = v.t.length - 1
        const s = v.series[key]
        const segs = viewSegments(v)
        return (
          <div className="stack">
            <div className="tile-grid">
              {TILE_VARS.map((k) => {
                const sk = v.series[k], bad = outOfRange(sk, n)
                return (
                  <button key={k} className={'card tilebtn' + (k === key ? ' sel' : '')} onClick={() => setKey(k)}>
                    <span className="muted small">{sk.label}</span>
                    <b className={'num ' + (bad ? 'down' : '')}>{fmt(sk.values[n], 2)}<small className="muted"> {sk.unit}</small></b>
                    <Spark values={sk.values.slice(-120)} color={bad ? 'var(--critical)' : 'var(--teal)'} />
                  </button>
                )
              })}
            </div>
            <div className="card">
              <h2>{s.label} ({s.unit})
                <select className="sel" value={key} onChange={(e) => setKey(e.target.value)}>
                  {Object.entries(v.series).map(([k, x]) => <option key={k} value={k}>{x.label}</option>)}
                </select>
              </h2>
              <SeriesChart rows={bandRows(v, key)} unit={s.unit} label={s.label} segments={segs}
                safe={[s.safe_lo, s.safe_hi]} height={320} dots={s.kind === 'offline'} />
              <p className="foot">{s.note ?? ''}{s.safe_source && ` 안전 범위 출처: ${s.safe_source}.`}{s.kind === 'offline' && ` Offline 측정, ${s.delay_h} h 뒤에 결과가 나옵니다.`}</p>
            </div>
            <div className="card">
              <h2>최근 이벤트</h2>
              <EventList events={v.events} />
            </div>
          </div>
        )
      }}</Load>
    </>
  )
}

function EventList({ events, sel, onPick }) {
  if (!events.length) return <p className="empty">감지된 이상 이벤트가 없습니다.</p>
  return (
    <ul className="events">
      {events.map((e, i) => (
        <li key={i} className={sel === i ? 'sel' : ''} onClick={onPick && (() => onPick(i))} style={{ cursor: onPick ? 'pointer' : undefined }}>
          <span className="num muted">{fmt(e.start)}–{fmt(e.end)} h</span>
          <span className="msg"><b>{e.label}</b> · {e.message}</span>
          <Pill state={e.grade === 'CRITICAL' ? 'anomaly' : 'caution'}>{e.grade}</Pill>
        </li>
      ))}
    </ul>
  )
}

// --- Risk analysis -------------------------------------------------------------------------------

export function Risk({ id, t }) {
  const view = useApi(withT(`/batches/${id}/view`, t))
  const [sel, setSel] = useState(null)
  useEffect(() => setSel(null), [id, t])
  return (
    <>
      <PageHead title="위험도 및 이상 감지" sub="Risk Analysis · 위험 점수 R(t) 0–100, 경계값을 1 h 이상 넘으면 경보" />
      <Load s={view} height={300}>{(v) => {
        const [warn, crit] = v.risk.cuts
        const rows = v.t.map((t, i) => ({ t, R: v.risk.R[i] }))
        const e = sel == null ? null : v.events[sel]
        const es = e && v.series[e.var]
        return (
          <div className="stack">
            <div className="grid">
              <div className="card span-4">
                <h2>현재 위험도</h2>
                <Gauge score={v.risk.R.at(-1)} cuts={v.risk.cuts} state={GRADE_STATE[v.risk.grade.at(-1)]} />
                <p className="explain">{v.explanation.text}</p>
                {v.risk.alarm_top3.length > 0 && <p className="foot center">경보 구간 주요 원인: {v.risk.alarm_top3.map((c) => `${v.series[c.var]?.label ?? c.var} ${fmt(c.share_pct, 0)}%`).join(', ')}</p>}
              </div>
              <div className="card span-8">
                <h2>위험도 추이 Risk Trend</h2>
                <ResponsiveContainer width="100%" height={260}>
                  <ComposedChart data={rows} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                    <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} tick={tick} tickFormatter={(x) => `${Math.round(x)}h`} stroke="var(--border)" />
                    <YAxis domain={[0, 100]} tick={tick} stroke="var(--border)" />
                    <ReferenceArea y1={crit} y2={100} fill="var(--critical)" fillOpacity={0.06} />
                    <ReferenceArea y1={warn} y2={crit} fill="var(--warning)" fillOpacity={0.07} />
                    {v.events.map((ev, i) => (
                      <ReferenceArea key={i} x1={ev.start} x2={ev.end} fill={ev.grade === 'CRITICAL' ? 'var(--critical)' : 'var(--warning)'}
                        fillOpacity={sel === i ? 0.3 : 0.12} onClick={() => setSel(i)} style={{ cursor: 'pointer' }} />
                    ))}
                    <ReferenceLine y={warn} stroke="var(--warning)" strokeDasharray="4 3" />
                    <ReferenceLine y={crit} stroke="var(--critical)" strokeDasharray="4 3" />
                    <Tooltip contentStyle={tip} labelFormatter={(x) => `${fmt(x)} h`} formatter={(x) => [fmt(x), 'R(t)']} />
                    <Area dataKey="R" stroke="none" fill="var(--primary)" fillOpacity={0.08} isAnimationActive={false} />
                    <Line dataKey="R" stroke="var(--primary)" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </ComposedChart>
                </ResponsiveContainer>
                <p className="foot">주의 ≥{warn} · 위험 ≥{crit} · 정상 배치(학습)의 오경보 비율로 정한 경계값입니다.</p>
              </div>
            </div>
            <div className="grid">
              <div className="card span-4">
                <h2>감지된 이상 이벤트 ({v.events.length})</h2>
                <EventList events={v.events} sel={sel} onPick={setSel} />
              </div>
              <div className="card span-8">
                <h2>{e ? `이상 구간 상세 · ${es.label}` : '이상 구간 상세'}</h2>
                {e ? (
                  <>
                    <SeriesChart rows={bandRows(v, e.var, e.start - 4, e.end + 4)} unit={es.unit} label={es.label}
                      segments={[{ start: e.start, end: e.end, state: e.grade === 'CRITICAL' ? 'anomaly' : 'caution' }]}
                      safe={[es.safe_lo, es.safe_hi]} height={280} dots={es.kind === 'offline'} />
                    <p className="foot">{e.message} · 기준: {e.source === 'band' ? '정상 밴드' : '안전 범위'} · 최대 이탈값 {fmt(e.peak, 3)} {e.unit}</p>
                  </>
                ) : <p className="empty">왼쪽 목록이나 차트에서 이상 구간을 선택하세요.</p>}
              </div>
            </div>
            <div className="card">
              <h2>안전 범위를 지켰다면? Rule What-if</h2>
              <p className="muted small">변수 하나를 안전 범위 안으로 되돌렸을 때 CRITICAL 시점이 얼마나 줄어드는지 (전체: {v.whatif.rule_all.critical_before} → {v.whatif.rule_all.critical_after}점, −{fmt(v.whatif.rule_all.reduction_pct, 0)}%)</p>
              <table className="tbl">
                <thead><tr><th>변수</th><th>CRITICAL 전</th><th>후</th><th>감소</th></tr></thead>
                <tbody>
                  {v.whatif.rule_by_var.filter((r) => r.critical_before > 0).map((r) => (
                    <tr key={r.var}><td>{v.series[r.var]?.label ?? r.var}</td><td className="num">{r.critical_before}</td><td className="num">{r.critical_after}</td>
                      <td className="num"><b className={r.reduction_pct > 0 ? 'up' : ''}>{fmt(r.reduction_pct, 0)}%</b></td></tr>
                  ))}
                </tbody>
              </table>
              <p className="foot">{v.whatif.rule_note}</p>
            </div>
          </div>
        )
      }}</Load>
    </>
  )
}

// --- Prediction ----------------------------------------------------------------------------------

function Stat({ label, value, unit, tone }) {
  return (
    <div className="card">
      <span className="muted small">{label}</span>
      <div className="big num" style={{ color: tone }}>{value}{unit && <small>{unit}</small>}</div>
    </div>
  )
}

export function Prediction({ id, t, fullH }) {
  const now = t ?? fullH
  const detail = useApi(withT(`/batches/${id}`, t))
  const forecast = useApi(now > 0 && `/batches/${id}/forecast?t=${now}`)
  const view = useApi(withT(`/batches/${id}/view`, t))
  return (
    <>
      <PageHead title="예측 분석" sub="8-hour Concentration Forecast · 실선은 실측(오프라인 측정, 지연 반영), 점선은 예측" />
      <Load s={forecast} height={300}>{(f) => (
        <Load s={detail}>{(d) => {
          const cur = f.last_measured.value, last = f.path.at(-1)
          const chg = ((last.p - cur) / cur) * 100
          const obs = d.series.filter((p) => p.t >= now - 48).map((p) => ({ t: p.t, obs: p.penicillin }))
          const rows = [...obs, { t: now, obs: obs.at(-1)?.obs, pred: obs.at(-1)?.obs }, ...f.path.map((p) => ({ t: p.t, pred: p.p, band: [p.lo, p.hi] }))]
          return (
            <div className="stack">
              <div className="grid">
                <div className="span-3"><Stat label={`현재 농도 (${f.last_measured.age_h} h 전 측정)`} value={fmt(cur, 2)} unit={f.unit} /></div>
                <div className="span-3"><Stat label="8시간 후 예측" value={fmt(last.p, 2)} unit={f.unit} tone="var(--primary)" /></div>
                <div className="span-3"><Stat label="변화율" value={`${chg > 0 ? '+' : ''}${fmt(chg)}`} unit="%" tone={chg >= 0 ? 'var(--stable)' : 'var(--critical)'} /></div>
                <div className="span-3"><Stat label="80% 예측 구간" value={`${fmt(last.lo, 2)}–${fmt(last.hi, 2)}`} unit={f.unit} /></div>
              </div>
              <div className="card">
                <h2>농도 예측 {f.label} ({f.unit})</h2>
                <ResponsiveContainer width="100%" height={340}>
                  <ComposedChart data={rows} margin={{ top: 16, right: 8, left: -10, bottom: 0 }}>
                    <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} tick={tick} tickFormatter={(x) => `${Math.round(x)}h`} stroke="var(--border)" />
                    <YAxis domain={['auto', 'auto']} tick={tick} stroke="var(--border)" />
                    <ReferenceArea x1={now} x2={last.t} fill="var(--primary)" fillOpacity={0.04} />
                    <Tooltip contentStyle={tip} labelFormatter={(x) => `${fmt(x)} h`} formatter={(x, n) => [Array.isArray(x) ? x.map((y) => fmt(y, 2)).join(' – ') : fmt(x, 2), n]} />
                    <Legend iconType="plainline" wrapperStyle={{ fontSize: 11 }} />
                    <Area dataKey="band" name="80% 예측 구간" stroke="none" fill="var(--teal)" fillOpacity={0.22} isAnimationActive={false} />
                    <Line dataKey="obs" name="실측값" stroke="var(--foreground)" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line dataKey="pred" name="예측값" stroke="var(--primary)" strokeWidth={2.2} strokeDasharray="6 4" dot={false} isAnimationActive={false} />
                    <ReferenceLine x={now} stroke="var(--foreground)" strokeDasharray="2 3" label={{ value: '현재', position: 'top', fontSize: 11 }} />
                  </ComposedChart>
                </ResponsiveContainer>
                <p className="foot">{f.note} 음영 구간은 모델 예측 불확실성이며 정상 운전 범위가 아닙니다.</p>
              </div>
              <div className="card">
                <h2>최종 수율 예측 Final Yield <span className="muted small">각 시점까지의 데이터만 사용</span></h2>
                <Load s={view}>{(v) => <YieldTrend v={v} />}</Load>
              </div>
            </div>
          )
        }}</Load>
      )}</Load>
    </>
  )
}

function YieldTrend({ v }) {
  const tr = v.pred_trend
  if (!tr.length) return <p className="empty">첫 예측 시점({v.t_cut > 46 ? 46 : v.t_cut} h) 전입니다.</p>
  const rows = tr.map((p) => ({ t: p.t_cut, y: p.y_pred, band: [p.y_pred_lo, p.y_pred_hi], model: p.model }))
  const lastP = tr.at(-1)
  return (
    <>
      <p className="num" style={{ margin: '0 0 8px' }}>
        <b className="big">{fmt(lastP.y_pred, 0)} kg</b>
        <span className="muted small"> 80% 구간 {fmt(lastP.y_pred_lo, 0)}–{fmt(lastP.y_pred_hi, 0)} kg · {lastP.t_cut} h 모델({lastP.model})</span>
        {v.y_true_kg != null && <span className="small"> · 실제 {fmt(v.y_true_kg, 0)} kg</span>}
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} ticks={rows.map((r) => r.t)} tick={tick} tickFormatter={(x) => `${x}h`} stroke="var(--border)" />
          <YAxis domain={['auto', 'auto']} tick={tick} stroke="var(--border)" tickFormatter={(x) => fmt(x, 0)} width={60} />
          <Tooltip contentStyle={tip} labelFormatter={(x) => `${x} h 시점 예측`} formatter={(x, n) => [Array.isArray(x) ? x.map((y) => fmt(y, 0)).join(' – ') + ' kg' : fmt(x, 0) + ' kg', n]} />
          <Area dataKey="band" name="80% 구간" stroke="none" fill="var(--teal)" fillOpacity={0.22} isAnimationActive={false} />
          <Line dataKey="y" name="예측 최종 수율" stroke="var(--primary)" strokeWidth={2} isAnimationActive={false} />
          {v.y_true_kg != null && <ReferenceLine y={v.y_true_kg} stroke="var(--foreground)" strokeDasharray="4 3" label={{ value: '실제', position: 'right', fontSize: 11 }} />}
        </ComposedChart>
      </ResponsiveContainer>
      <p className="foot">{v.target.text}. {v.t_cut} h 이전은 ridge 조기 추정, 이후는 XGBoost입니다.</p>
    </>
  )
}

// --- Root cause ----------------------------------------------------------------------------------

export function RootCause({ id, t, model }) {
  const view = useApi(withT(`/batches/${id}/view`, t))
  const cc = model.cross_check
  return (
    <>
      <PageHead title="원인 분석" sub="Explainable AI · SHAP + Permutation Importance 교차 검증 · 결정트리 규칙" />
      <div className="note">ⓘ 기여도는 모델이 예측을 내릴 때 각 변수가 미친 영향을 나타냅니다. 물리적 인과관계의 증거로 해석하지 마세요.</div>
      <Load s={view} height={300}>{(v) => {
        const all = v.shap_all
        if (!all.length) return <div className="card empty">첫 예측 시점 전이라 기여도가 아직 없습니다.</div>
        const top = all.slice(0, 10).map((d) => ({ name: d.label, v: d.shap, f: d.feature }))
        const yhat = v.y_pred_kg ?? v.pred_latest_kg
        return (
          <div className="grid">
            <div className="card span-5">
              <h2>Top 3 영향 변수</h2>
              <ol className="top3">
                {all.slice(0, 3).map((d, i) => (
                  <li key={d.feature}>
                    <span className="badge">{i + 1}</span>
                    <div><b>{d.label}</b>
                      <p className="muted small">현재 값 {fmt(d.value, 2)} · 예측 수율을 {d.shap > 0 ? '높이는' : '낮추는'} 방향 · {d.shap > 0 ? '+' : ''}{fmt(d.shap, 0)} kg</p></div>
                  </li>
                ))}
              </ol>
              <p className="foot">기준값 {fmt(v.shap_base, 0)} kg + 기여도 합 = 예측 {fmt(yhat, 0)} kg ({v.shap_t_cut} h 모델)</p>
            </div>
            <div className="card span-7">
              <h2>Feature Attribution · SHAP (kg)</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={top} layout="vertical" margin={{ top: 0, right: 16, left: 0, bottom: 0 }}>
                  <XAxis type="number" tick={tick} stroke="var(--border)" tickFormatter={(x) => fmt(x, 0)} />
                  <YAxis type="category" dataKey="name" width={220} tick={{ ...tick, fontSize: 10 }} stroke="var(--border)" />
                  <Tooltip contentStyle={tip} formatter={(x) => [`${fmt(x, 0)} kg`, 'SHAP']} />
                  <ReferenceLine x={0} stroke="var(--muted-foreground)" />
                  <Bar dataKey="v" isAnimationActive={false} radius={4}>
                    {top.map((d) => <Cell key={d.f} fill={d.v > 0 ? 'var(--stable)' : 'var(--critical)'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="card span-8">
              <h2>설명 방법 교차 검증 Cross-check <span className="muted small">학습 배치 전체, 폴드별 held-out</span></h2>
              <p><b className="big num">{fmt(cc.overlap.top3 * 100, 0)}%</b> <span className="muted">Top 3 일치율 (SHAP vs Permutation) · Top 5 {fmt(cc.overlap.top5 * 100, 0)}%</span></p>
              <table className="tbl">
                <thead><tr><th>변수</th><th>SHAP 순위</th><th>Permutation 순위</th><th>|SHAP| kg</th><th>방향 (모델 / 데이터)</th></tr></thead>
                <tbody>
                  {cc.top.map((r) => (
                    <tr key={r.feature}><td title={r.feature}>{r.label}</td><td className="num">{r.shap_rank}</td><td className="num">{r.perm_rank}</td>
                      <td className="num">{fmt(r.shap_kg, 0)}</td>
                      <td><span className={r.shap_direction === r.data_direction ? 'up' : 'down'}>{r.shap_direction === r.data_direction ? '✓ Agree' : '✗ Disagree'}</span>
                        <span className="muted small"> ({r.shap_direction > 0 ? '+' : '−'} / {r.data_direction > 0 ? '+' : '−'})</span></td></tr>
                  ))}
                </tbody>
              </table>
              <p className="foot">{cc.note}</p>
            </div>
            <div className="card span-4">
              <h2>결정트리 규칙</h2>
              {v.rule ? (
                <>
                  <ul className="conds">{v.rule.conditions.map((c) => <li key={c} className="num">{c}</li>)}</ul>
                  <p>→ 평균 수율 <b className="num">{fmt(v.rule.mean_y, 0)} kg</b> <span className="muted small">({v.rule.n}개 배치)</span></p>
                  <p className="foot">이 배치가 속한 잎입니다. 트리 R² 학습 {fmt(v.rule.tree_r2.train, 2)} · CV {fmt(v.rule.tree_r2.train_cv, 2)}{v.rule.reference_only && ' · 참고용'}</p>
                </>
              ) : <p className="empty">{v.t_cut} h 이후에 표시됩니다.</p>}
            </div>
          </div>
        )
      }}</Load>
    </>
  )
}

// --- What-if -------------------------------------------------------------------------------------

export function WhatIf({ id, t }) {
  const view = useApi(withT(`/batches/${id}/view`, t))
  return (
    <>
      <PageHead title="What-if 시뮬레이션" sub="전반부(0 – t_cut) 운전 총량을 바꿨을 때 예측 최종 수율 · 실제 공정에는 적용되지 않습니다" />
      <Load s={view} height={300}>{(v) => v.whatif.sliders.length
        ? <WhatIfBody key={`${id}-${t}`} id={id} v={v} />
        : <div className="card empty">What-if는 {v.t_cut} h 이후에 사용할 수 있습니다 (모델이 0–{v.t_cut} h 데이터로 예측).</div>}</Load>
    </>
  )
}

function WhatIfBody({ id, v }) {
  const sl = v.whatif.sliders
  const init = Object.fromEntries(sl.map((s) => [s.feature, s.value]))
  const label = Object.fromEntries(sl.map((s) => [s.feature, s.label]))
  const [vals, setVals] = useState(init)
  const [focus, setFocus] = useState(sl[0].feature)
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const changed = sl.filter((s) => Math.abs(vals[s.feature] - s.value) > 1e-9)
  const run = () => {
    setBusy(true); setErr(null)
    get('/simulate', { batch_id: id, changes: Object.fromEntries(changed.map((s) => [s.feature, vals[s.feature] / s.value - 1])) })
      .then(setRes, setErr).finally(() => setBusy(false))
  }
  const fs = sl.find((s) => s.feature === focus)
  const curve = fs.curve.x.map((x, i) => ({ x, y: fs.curve.y[i] }))
  const yAt = (x) => curve.reduce((a, b) => (Math.abs(b.x - x) < Math.abs(a.x - x) ? b : a)).y
  return (
    <div className="grid">
      <div className="card span-7">
        <h2>운전 조건 조정</h2>
        {sl.map((s) => {
          const x = vals[s.feature], moved = Math.abs(x - s.value) > 1e-9
          return (
            <div key={s.feature} className={'slider' + (focus === s.feature ? ' focus' : '')} onPointerDown={() => setFocus(s.feature)}>
              <div className="row between small">
                <b>{s.label}</b>
                <span className="num"><span className="muted">원래 {fmt(s.value, 2)}</span> <span className={'val' + (moved ? ' moved' : '')}>{fmt(x, 2)}</span>
                  {moved && <span className="muted"> ({x > s.value ? '+' : ''}{fmt((x / s.value - 1) * 100)}%)</span>}</span>
              </div>
              <input type="range" min={s.min} max={s.max} step={(s.max - s.min) / 200} value={x}
                onChange={(e) => setVals({ ...vals, [s.feature]: +e.target.value })} onFocus={() => setFocus(s.feature)} />
              <div className="row between muted tiny"><span>{fmt(s.min, 1)}</span><span>학습 배치 범위</span><span>{fmt(s.max, 1)}</span></div>
            </div>
          )
        })}
        <div className="row gap">
          <button className="btn" onClick={() => { setVals(init); setRes(null) }}>↺ Reset</button>
          <button className="btn primary" onClick={run} disabled={busy || !changed.length}>{busy ? '계산 중…' : '▶ Run Simulation'}</button>
        </div>
        <p className="foot">{v.whatif.note}</p>
      </div>
      <div className="span-5 stack">
        <div className="card">
          <h2>예측 결과 비교</h2>
          {err && <p className="down small">{err.message}</p>}
          {!res && !err && <p className="empty">조건을 조정한 뒤 Run Simulation을 눌러주세요.</p>}
          {res && (
            <>
              <div className="yield-row big-row">
                <span className="muted">예상 최종 수율</span>
                <span className="num"><span className="muted">기존 {fmt(res.y_pred_before, 0)}</span> → <b className={res.delta >= 0 ? 'up' : 'down'}>{fmt(res.y_pred_after, 0)} kg</b></span>
              </div>
              <p className={'num center ' + (res.delta >= 0 ? 'up' : 'down')}><b>{res.delta >= 0 ? '+' : ''}{fmt(res.delta, 0)} kg</b></p>
              <table className="tbl">
                <thead><tr><th>입력</th><th>전</th><th>후</th></tr></thead>
                <tbody>{Object.entries(res.inputs).map(([f, x]) => (
                  <tr key={f}><td title={f}>{label[f] ?? f}{x.linked_to && <span className="muted tiny"> ({label[x.linked_to]} 연동)</span>}</td><td className="num">{fmt(x.before, 2)}</td><td className="num">{fmt(x.after, 2)}</td></tr>
                ))}</tbody>
              </table>
              <p className="muted small">변경 후 SHAP Top 3: {res.top3.map((d) => `${d.label} (${d.shap > 0 ? '+' : ''}${fmt(d.shap, 0)})`).join(', ')}</p>
            </>
          )}
        </div>
        <div className="card">
          <h2>반응 곡선 · {fs.label}</h2>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={curve} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} tick={tick} tickFormatter={(x) => fmt(x, 0)} stroke="var(--border)" />
              <YAxis domain={['auto', 'auto']} tick={tick} tickFormatter={(x) => fmt(x, 0)} width={56} stroke="var(--border)" />
              <Tooltip contentStyle={tip} labelFormatter={(x) => fmt(x, 2)} formatter={(x) => [`${fmt(x, 0)} kg`, '예측 수율']} />
              <Line dataKey="y" type="stepAfter" stroke="var(--primary)" strokeWidth={2} dot={false} isAnimationActive={false} />
              <ReferenceDot x={fs.value} y={yAt(fs.value)} r={5} fill="var(--muted-foreground)" stroke="#fff" />
              <ReferenceDot x={vals[focus]} y={yAt(vals[focus])} r={6} fill="var(--primary)" stroke="#fff" />
            </LineChart>
          </ResponsiveContainer>
          <p className="foot">다른 변수는 그대로 두고 이 변수만 움직였을 때의 예측입니다. 회색 = 원래 값, 파랑 = 조정 값. 트리 모델이라 계단 모양입니다.</p>
        </div>
      </div>
    </div>
  )
}

// --- Replay --------------------------------------------------------------------------------------

const REPLAY_VARS = ['temperature', 'DO', 'pH', 'penicillin']

export function Replay({ id }) {
  const detail = useApi(`/batches/${id}`)
  const events = useApi(`/batches/${id}/events`)
  return (
    <>
      <PageHead title="배치 리플레이" sub="Batch Replay · 모든 차트가 선택한 시각으로 동기화됩니다" />
      <Load s={detail} height={300}>{(d) => <ReplayBody key={id} d={d} events={events.data ?? []} />}</Load>
    </>
  )
}

function ReplayBody({ d, events }) {
  const end = d.series.at(-1).t
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(2)
  useEffect(() => {
    if (!playing) return
    const h = setInterval(() => setCur((c) => {
      if (c >= end) { setPlaying(false); return end }
      return Math.min(end, c + 0.5)
    }), 300 / speed)
    return () => clearInterval(h)
  }, [playing, speed, end])
  const row = d.series.find((r) => r.t >= cur) ?? d.series.at(-1)
  const shown = useMemo(() => d.series.filter((r) => r.t <= cur), [d.series, cur])
  const segs = d.segments.filter((s) => s.start <= cur).map((s) => ({ ...s, end: Math.min(s.end, cur) }))
  return (
    <div className="stack">
      <div className="card">
        <div className="row gap wrap">
          <button className="play" onClick={() => { if (cur >= end) setCur(0); setPlaying(!playing) }} aria-label={playing ? '일시정지' : '재생'}>{playing ? '❚❚' : '▶'}</button>
          <button className="btn" onClick={() => setCur(0)} aria-label="처음으로">⏮</button>
          <div><span className="muted small">재생 시각</span><div className="num big">+{fmt(cur)} h</div></div>
          <select className="sel push" value={speed} onChange={(e) => setSpeed(+e.target.value)}>
            {[1, 2, 4, 8].map((s) => <option key={s} value={s}>{s}×</option>)}
          </select>
        </div>
        <div className="scrub">
          <div className="marks">
            {events.map((e, i) => <i key={i} style={{ left: `${(e.start / end) * 100}%`, width: `${Math.max(((e.end - e.start) / end) * 100, 0.4)}%`, background: e.grade === 'CRITICAL' ? 'var(--critical)' : 'var(--warning)' }} />)}
          </div>
          <input type="range" min={0} max={end} step={0.2} value={cur} onChange={(e) => { setCur(+e.target.value); setPlaying(false) }} />
        </div>
      </div>
      <div className="grid">
        <div className="card span-3">
          <h2>선택 시각 위험도</h2>
          <Gauge score={row.score} cuts={d.scoreCuts} state={row.state} size={150} />
          <div className="mini-grid">
            {REPLAY_VARS.map((k) => (
              <div key={k}><span className="muted tiny">{d.seriesMeta[k].label}</span><b className="num small">{fmt(row[k], 2)} {d.seriesMeta[k].unit}</b></div>
            ))}
          </div>
          <p className="foot center">상태 <Pill state={row.state} /></p>
        </div>
        <div className="span-9 replay-grid">
          {REPLAY_VARS.map((k) => (
            <div className="card" key={k}>
              <h2>{d.seriesMeta[k].label} ({d.seriesMeta[k].unit})</h2>
              <SeriesChart rows={shown.map((r) => ({ t: r.t, v: r[k] }))} unit={d.seriesMeta[k].unit} label={d.seriesMeta[k].label}
                segments={segs} cursor={cur} domain={[0, end]} height={170} />
            </div>
          ))}
        </div>
      </div>
      <div className="card">
        <h2>이상 이벤트 마커</h2>
        {events.length === 0 && <p className="empty">이벤트 없음</p>}
        <ul className="events">
          {events.map((e, i) => (
            <li key={i} onClick={() => { setCur(e.start); setPlaying(false) }} style={{ cursor: 'pointer', opacity: e.start <= cur ? 1 : 0.45 }}>
              <span className="num muted">{fmt(e.start)} h</span>
              <span className="msg"><b>{e.label}</b> · {e.message}</span>
              <Pill state={e.grade === 'CRITICAL' ? 'anomaly' : 'caution'}>{e.grade}</Pill>
            </li>
          ))}
        </ul>
        <p className="foot">재생된 데이터 포인트: {shown.length}/{d.series.length}</p>
      </div>
    </div>
  )
}

// --- Optimization --------------------------------------------------------------------------------

export function Optimization({ id }) {
  const [mc, setMc] = useState(0.2)
  const opt = useApi(`/batches/${id}/optimize`, { max_change: mc })
  const view = useApi(`/batches/${id}/view`)
  return (
    <>
      <PageHead title="운전조건 최적화" sub="Optimization · 목표: 예측 최종 수율 최대화 · 추천값은 참고용이며 실제 발효 공정에 적용되지 않았습니다"
        right={<label className="small muted">변경 허용 폭&nbsp;
          <select className="sel" value={mc} onChange={(e) => setMc(+e.target.value)}>
            {[0.1, 0.2, 0.3, 0.5].map((x) => <option key={x} value={x}>±{x * 100}%</option>)}
          </select></label>} />
      <Load s={opt} height={300}>{(o) => (
        <div className="grid">
          <div className="card span-8">
            <h2>현재 vs 추천 운전조건</h2>
            <Load s={view}>{(v) => {
              const b = Object.fromEntries(v.whatif.sliders.map((s) => [s.feature, [s.min, s.max]]))
              return Object.keys(o.best).map((f) => {
                const cur = o.current[f], lo = Math.max(b[f][0], cur * (1 - mc)), hi = Math.min(b[f][1], cur * (1 + mc))
                const pos = (x) => `${((x - lo) / (hi - lo || 1)) * 100}%`
                return (
                  <div key={f} className="optrow">
                    <div className="row between small"><b>{o.labels[f]}</b>
                      <span className="num"><span className="muted">{fmt(cur, 2)}</span> → <b className="pri">{fmt(o.best[f], 2)}</b>
                        <span className={o.change_pct[f] >= 0 ? 'up' : 'down'}> ({o.change_pct[f] >= 0 ? '+' : ''}{fmt(o.change_pct[f])}%)</span></span></div>
                    <div className="track"><i className="cur" style={{ left: pos(cur) }} /><i className="best" style={{ left: pos(o.best[f]) }} /></div>
                    <div className="row between muted tiny"><span>제약 {fmt(lo, 1)}</span><span>{fmt(hi, 1)}</span></div>
                  </div>
                )
              })
            }}</Load>
            <div className="row gap muted tiny"><span><i className="legend cur" /> 현재</span><span><i className="legend best" /> 추천</span></div>
          </div>
          <div className="span-4 stack">
            <div className="card">
              <h2>예상 개선 효과</h2>
              <div className="yield-row"><span className="muted">예측 최종 수율</span><span className="num">{fmt(o.y_pred_current, 0)} → <b>{fmt(o.y_pred_best, 0)} kg</b></span></div>
              <p className="big num up" style={{ margin: '10px 0 0' }}>+{fmt(o.gain_kg, 0)}<small>kg</small></p>
              <p className="muted small">{o.models_agree}/{o.n_models} 폴드 모델이 개선을 예측 · 모델 간 편차 {fmt(o.spread_current_kg, 0)} → {fmt(o.spread_best_kg, 0)} kg</p>
              <ResponsiveContainer width="100%" height={120}>
                <BarChart data={o.gain_by_model_kg.map((g, i) => ({ m: `fold ${i + 1}`, g }))} margin={{ top: 8, right: 0, left: -20, bottom: 0 }}>
                  <XAxis dataKey="m" tick={tick} stroke="var(--border)" /><YAxis tick={tick} stroke="var(--border)" />
                  <Tooltip contentStyle={tip} formatter={(x) => [`${fmt(x, 0)} kg`, '개선']} />
                  <ReferenceLine y={0} stroke="var(--muted-foreground)" />
                  <Bar dataKey="g" isAnimationActive={false} radius={4}>
                    {o.gain_by_model_kg.map((g, i) => <Cell key={i} fill={g >= 0 ? 'var(--stable)' : 'var(--critical)'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="card">
              <h2>참고 사항</h2>
              <p className="foot" style={{ marginTop: 0 }}>{o.note}</p>
            </div>
          </div>
        </div>
      )}</Load>
    </>
  )
}
