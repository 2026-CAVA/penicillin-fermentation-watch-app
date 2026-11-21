import { useEffect, useState } from 'react'
import { STATE, useApi } from './lib.jsx'
import * as P from './pages.jsx'

const NAV = [
  ['overview', 'Overview', '개요', P.Overview], ['batches', 'Batch Selection', '배치 선택', P.Batches],
  ['monitoring', 'Process Monitoring', '공정 모니터링', P.Monitoring], ['risk', 'Risk Analysis', '이상 감지', P.Risk],
  ['prediction', 'Prediction', '예측 분석', P.Prediction], ['root-cause', 'Root Cause', '원인 분석', P.RootCause],
  ['what-if', 'What-if Simulation', '시뮬레이션', P.WhatIf], ['replay', 'Batch Replay', '배치 리플레이', P.Replay],
  ['optimization', 'Optimization', '최적화', P.Optimization],
]
const pageOf = () => NAV.find((n) => '#/' + n[0] === location.hash) ?? NAV[0]

export default function App() {
  const model = useApi('/model')
  const list = useApi('/batches')
  const [id, setId] = useState(null)
  const [t, setT] = useState(null) // null = whole batch; a number = "running" at t hours (shared by every page)
  const [tDraft, setTDraft] = useState(null)
  const [page, setPage] = useState(pageOf)

  useEffect(() => {
    const on = () => { setPage(pageOf()); window.scrollTo(0, 0) }
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [])
  useEffect(() => {
    if (id == null && model.data && list.data) setId(list.data[0].id)
  }, [id, model.data, list.data])

  const batches = list.data ?? []
  const fullH = batches.find((b) => b.id === id)?.elapsedH ?? 0
  const pick = (v) => { setId(+v); setT(null); setTDraft(null) }
  const commitT = () => setT(tDraft == null || tDraft >= fullH ? null : tDraft)
  const go = (key) => { location.hash = '/' + key }
  const [key, , , Page] = page
  const error = model.error || list.error

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="logo">CAV<span>A</span></div>
        <div className="tagline">Fermentation Process Analytics &amp; Optimization</div>
        {NAV.map(([k, en, ko]) => (
          <a key={k} href={'#/' + k} className={'nav' + (k === key ? ' active' : '')}>{en} <small>{ko}</small></a>
        ))}
        <div className="pipeline">Monitor → Detect → Predict → Explain → Optimize</div>
      </aside>

      <div>
        <div className="topbar">
          <div className="picker">
            <label>현재 배치
              <select value={id ?? ''} onChange={(e) => pick(e.target.value)}>
                {batches.map((b) => <option key={b.id} value={b.id}>Batch #{b.id} · {STATE[b.state]}</option>)}
              </select>
            </label>
            {key !== 'replay' && <>
              <label>시점 t = {tDraft ?? t ?? fullH} h
                <input type="range" min={1} max={fullH} step={1} value={tDraft ?? t ?? fullH}
                  onChange={(e) => setTDraft(+e.target.value)} onPointerUp={commitT} onKeyUp={commitT} />
              </label>
              <button onClick={() => { setT(null); setTDraft(null) }}>끝까지</button>
              <span className={'pill ' + (t == null ? 'normal' : 'info')}>{t == null ? 'Completed' : 'Running'}</span>
            </>}
          </div>
        </div>

        <main>
          {error && <div className="card anomaly">API 오류: {error.message} — <code>python -m main serve</code>가 켜져 있는지 확인하세요.</div>}
          {id != null && model.data && <Page id={id} t={t} fullH={fullH} model={model.data} batches={batches} setId={pick} go={go} />}
        </main>
      </div>
    </div>
  )
}
