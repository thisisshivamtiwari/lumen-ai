"""False-positive / false-negative report for the V1 grouped out-of-fold validation run.

Reads only saved V1 OOF scores, training labels and label-free payment fields. Annotation
columns (family, difficulty, scenario_id, cohort_id) are used for reporting, never as predictors.
Run from the project root: .venv/bin/python outputs/fp_fn_report/source/fp_fn_report.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/fp_fn_report'
sys.path.insert(0, str(ROOT / 'data/training'))
from payments import load_payments  # noqa: E402

TYPE = {'fan_out_in': 'Scatter-gather', 'multi_stage': 'Scatter-gather',
        'stack': 'Layered stack', 'chain': 'Layered stack', 'distributed_relay': 'Layered stack',
        'bipartite': 'Bipartite', 'aggregate_relay': 'Gather-scatter (relay hubs)', 'cycle': 'Cycle',
        'fan_out': 'Fan-out', 'random': 'Irregular network', 'fan_in': 'Fan-in',
        'background': 'Background (no scenario)', 'ambiguous': 'Misc (ambiguous)'}
ORDER = list(dict.fromkeys(TYPE.values()))
DIFF = ['easy', 'medium', 'hard']


def md(frame, floats=1):
    """Plain Markdown table without extra dependencies."""
    cells = frame.astype(object).where(frame.notna(), '–')
    fmt = lambda v: f'{v:,.{floats}f}' if isinstance(v, float) else (f'{v:,}' if isinstance(v, (int, np.integer)) else str(v))
    rows = [[fmt(v) for v in r] for r in cells.itertuples(index=False)]
    head = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    return '\n'.join([head, '|' + '---|' * len(frame.columns)] + ['| ' + ' | '.join(r) + ' |' for r in rows])


def type_table(d, k):
    top = d['rank'] <= k
    pos, neg = d.is_suspicious == 1, d.is_suspicious == 0
    fp_total = int((top & neg).sum())
    rows = []
    for t in ORDER + ['Total']:
        g = d.type == t if t != 'Total' else pd.Series(True, index=d.index)
        n_pos, n_neg = int((g & pos).sum()), int((g & neg).sum())
        fp, tp = int((g & neg & top).sum()), int((g & pos & top).sum())
        rows.append({'Type': t, 'Suspicious': n_pos, 'Normal look-alikes': n_neg,
                     f'FP: normal in the top {k:,}': fp,
                     'Share of all FP': f'{fp / fp_total:.0%}' if fp_total else '–',
                     f'Suspicious in the top {k:,}': tp if n_pos else None,
                     'FN: suspicious below it': n_pos - tp if n_pos else None,
                     'FN share': f'{(n_pos - tp) / n_pos:.0%}' if n_pos else None,
                     'Median rank of suspicious': int(d.loc[g & pos, 'rank'].median()) if n_pos else None})
    table = pd.DataFrame(rows)
    for c in table.columns[5:]:
        if c not in ('FN share',):
            table[c] = table[c].astype('Int64')
    return table


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(ROOT / 'outputs/strategy3_v1/graph_oof.csv', usecols=['TRANSACTION_HASH', 'fold', 'score'])
    labels = pd.read_csv(ROOT / 'data/training/train/labels.csv')
    pay = pd.DataFrame(load_payments(ROOT / 'data/training/train'))[['TRANSACTION_HASH', 'TIMESTAMP', 'FROM_ADDRESS', 'TO_ADDRESS', 'VALUE']]
    feats = pd.read_parquet(ROOT / 'outputs/features_v1/train.parquet',
                            columns=['TRANSACTION_HASH', 'execution_legs', 'pair_count', 'flow_300s_two_sided_count', 'flow_86400s_two_sided_count'])
    d = labels.merge(oof, on='TRANSACTION_HASH', validate='1:1').merge(pay, on='TRANSACTION_HASH', validate='1:1').merge(feats, on='TRANSACTION_HASH', validate='1:1')
    assert len(d) == len(labels) == 40131 and d.score.notna().all()
    d['VALUE'] = d.VALUE.astype(float)
    d['type'] = d.family.map(TYPE)
    assert d.type.notna().all()
    # Portal convention: higher score first, ties broken by transaction hash ascending.
    d = d.sort_values(['score', 'TRANSACTION_HASH'], ascending=[False, True]).reset_index(drop=True)
    d['rank'] = np.arange(1, len(d) + 1)
    d['hours_from_start'] = ((d.TIMESTAMP - d.TIMESTAMP.min()) / 3600).astype(float).round(2)
    ap = average_precision_score(d.is_suspicious, d.score)
    k, budget = int(d.is_suspicious.sum()), int(np.ceil(0.01 * len(d)))
    pos, neg = d.is_suspicious == 1, d.is_suspicious == 0

    t1250, t402 = type_table(d, k), type_table(d, budget)
    t1250.to_csv(OUT / 'type_table_top1250.csv', index=False)
    t402.to_csv(OUT / 'type_table_top402.csv', index=False)

    # Family x difficulty: separates the hard relay tier hidden inside type names.
    fam = []
    for (f, diff), g in d[pos].groupby(['family', 'difficulty']):
        fam.append({'Family': f, 'Type': TYPE[f], 'Difficulty': diff, 'Suspicious': len(g),
                    f'In top {k:,}': int((g['rank'] <= k).sum()), f'In top {budget}': int((g['rank'] <= budget).sum()),
                    'Recall@1,250': f"{(g['rank'] <= k).mean():.0%}", 'Median rank': int(g['rank'].median()),
                    'Category AP': average_precision_score(np.r_[np.ones(len(g)), np.zeros(int(neg.sum()))], np.r_[g.score, d.loc[neg, 'score']])})
    fam = pd.DataFrame(fam).sort_values(['Difficulty', 'Suspicious'], ascending=[True, False])
    fam.to_csv(OUT / 'family_difficulty_table.csv', index=False)

    # False positives: rank bands and scenario concentration.
    fp = d[neg & (d['rank'] <= k)].copy()
    bands = pd.cut(fp['rank'], [0, 100, budget, k], labels=['1–100', f'101–{budget}', f'{budget + 1}–{k:,}'])
    fp_bands = pd.crosstab(fp.type, bands).reindex([t for t in ORDER if t in set(fp.type)])
    fp_bands['Total'] = fp_bands.sum(axis=1)
    fp_bands = fp_bands.sort_values('Total', ascending=False).reset_index().rename(columns={'type': 'Type'})
    fp['group'] = fp.scenario_id.fillna('background:' + fp.TRANSACTION_HASH.str[:8])
    fp_scen = (fp.groupby(['group', 'family', 'type'])
               .agg(fp_rows=('rank', 'size'), best_rank=('rank', 'min'), median_rank=('rank', 'median'))
               .reset_index().sort_values(['fp_rows', 'best_rank'], ascending=[False, True]))
    scen_size = d[neg].groupby('scenario_id').size()
    fp_scen['scenario_rows'] = fp_scen.group.map(scen_size).fillna(1).astype(int)
    fp_scen.to_csv(OUT / 'false_positive_scenarios.csv', index=False)
    wallets = pd.concat([fp.FROM_ADDRESS, fp.TO_ADDRESS]).value_counts()
    cols = ['rank', 'score', 'type', 'family', 'scenario_id', 'fold', 'hours_from_start', 'FROM_ADDRESS', 'TO_ADDRESS', 'VALUE',
            'execution_legs', 'pair_count', 'flow_300s_two_sided_count', 'flow_86400s_two_sided_count', 'TRANSACTION_HASH']
    fp[cols].to_csv(OUT / 'false_positives_top1250.csv', index=False)

    # False negatives: suspicious ranked below K, and scenario-level catch status.
    fn = d[pos & (d['rank'] > k)].copy()
    fn[cols[:4] + ['difficulty'] + cols[4:]].sort_values('rank', ascending=False).to_csv(OUT / 'false_negatives_below1250.csv', index=False)
    fn_bands = pd.crosstab(fn.type, pd.cut(fn['rank'], [k, 2500, 5000, 10000, len(d)],
                                           labels=[f'{k + 1:,}–2,500', '2,501–5,000', '5,001–10,000', f'10,001–{len(d):,}']))
    fn_bands = fn_bands.reindex([t for t in ORDER if t in set(fn.type)])
    fn_bands['Total'] = fn_bands.sum(axis=1)
    fn_bands = fn_bands.sort_values('Total', ascending=False).reset_index().rename(columns={'type': 'Type'})
    sp = d[pos & d.scenario_id.notna()].groupby(['scenario_id', 'family', 'type', 'difficulty']).agg(
        rows=('rank', 'size'), caught=('rank', lambda r: int((r <= k).sum())), best_rank=('rank', 'min'), worst_rank=('rank', 'max')).reset_index()
    sp['status'] = np.select([sp.caught == sp.rows, sp.caught > 0], ['fully caught', 'partly caught'], 'fully missed')
    sp.to_csv(OUT / 'suspicious_scenarios.csv', index=False)
    status = (sp.groupby(['type', 'status']).agg(scenarios=('rows', 'size'), rows=('rows', 'sum'), caught=('caught', 'sum'))
              .reset_index())
    status['missed_rows'] = status.rows - status.caught
    status_tab = status.pivot_table(index='type', columns='status', values='missed_rows', aggfunc='sum', fill_value=0)
    status_tab = status_tab.reindex(columns=['partly caught', 'fully missed'], fill_value=0)
    status_tab.columns = ['Missed rows in partly caught scenarios', 'Missed rows in fully missed scenarios']
    scen_counts = sp.pivot_table(index='type', columns='status', values='rows', aggfunc='size', fill_value=0).reindex(
        columns=['fully caught', 'partly caught', 'fully missed'], fill_value=0)
    scen_counts.columns = ['Scenarios fully caught', 'Scenarios partly caught', 'Scenarios fully missed']
    scen = scen_counts.join(status_tab).reindex([t for t in ORDER if t in set(sp.type)]).reset_index().rename(columns={'type': 'Type'})

    # Rank bands: every 500 rows to 5,000, then wider bands to the bottom (full 500-row bands in the CSV).
    def band_tables(edges):
        labels_ = [f'{lo + 1:,}–{hi:,}' for lo, hi in zip(edges[:-1], edges[1:])]
        band = pd.cut(d['rank'], edges, labels=labels_)
        sus = pd.crosstab(d.type[pos], band[pos]).reindex(index=[t for t in ORDER if t in set(d.type[pos])], columns=labels_, fill_value=0)
        nor = pd.crosstab(d.type[neg], band[neg]).reindex(index=[t for t in ORDER if t in set(d.type[neg])], columns=labels_, fill_value=0)
        sus.loc['Total suspicious'] = sus.sum()
        nor.loc['Total normal'] = nor.sum()
        size = band.value_counts().reindex(labels_)
        tp = sus.loc['Total suspicious']
        summary_ = pd.DataFrame({'Band': labels_, 'Rows': size.to_numpy(), 'Suspicious': tp.to_numpy(),
                                 'Normal (FP)': nor.loc['Total normal'].to_numpy(),
                                 'Band precision': [f'{v:.0%}' for v in tp.to_numpy() / size.to_numpy()],
                                 'Cumulative suspicious': tp.cumsum().to_numpy(),
                                 'Cumulative recall': [f'{v:.0%}' for v in tp.cumsum().to_numpy() / k],
                                 'Cumulative precision': [f'{v:.0%}' for v in tp.cumsum().to_numpy() / size.cumsum().to_numpy()]})
        return summary_, sus.reset_index().rename(columns={'type': 'Type', 'index': 'Type'}), nor.reset_index().rename(columns={'type': 'Type', 'index': 'Type'})

    view_edges = list(range(0, 5001, 500)) + [10000, 20000, len(d)]
    band_summary, band_sus, band_nor = band_tables(view_edges)
    full_summary, full_sus, full_nor = band_tables(list(range(0, len(d), 500)) + [len(d)])
    full_summary.to_csv(OUT / 'rank_bands_500_summary.csv', index=False)
    full_sus.to_csv(OUT / 'rank_bands_500_suspicious_by_type.csv', index=False)
    full_nor.to_csv(OUT / 'rank_bands_500_normal_by_type.csv', index=False)

    worst = fn[fn.difficulty != 'hard'].sort_values('rank', ascending=False).head(30)
    top_fp = fp.head(30)
    short = lambda f: f.assign(FROM_ADDRESS=f.FROM_ADDRESS.str[:8], TO_ADDRESS=f.TO_ADDRESS.str[:8],
                               scenario_id=f.scenario_id.fillna('–').str[:8], TRANSACTION_HASH=f.TRANSACTION_HASH.str[:10],
                               score=f.score.map('{:.4f}'.format), VALUE=f.VALUE.map('{:,.2f}'.format),
                               hours_from_start=f.hours_from_start.map('{:.1f}'.format),
                               execution_legs=f.execution_legs.astype(int), pair_count=f.pair_count.astype(int),
                               flow_300s_two_sided_count=f.flow_300s_two_sided_count.astype(int))
    partly_missed = int(scen['Missed rows in partly caught scenarios'].sum())
    fully_missed = int(scen['Missed rows in fully missed scenarios'].sum())
    medium = d[pos & (d.difficulty == 'medium')]
    whole_fp = fp_scen[(fp_scen.scenario_rows > 1) & (fp_scen.fp_rows >= 0.75 * fp_scen.scenario_rows)]
    top_budget_fp = t402.set_index('Type')[f'FP: normal in the top {budget}']
    findings = [
        '## Key findings', '',
        f'1. **{len(fp)} false positives and {len(fn)} false negatives at the top {k:,}** (precision = recall = {1 - len(fn) / k:.1%}). '
        f'Inside the portal budget of {budget}, there are {int(top_budget_fp["Total"])} false positives.',
        f'2. **False positives are whole legitimate scenarios, not isolated rows.** {len(whole_fp)} legitimate look-alike scenarios have ≥75% of their rows '
        'in the top 1,250 (for example, a 12-row bipartite scenario with all 12 rows flagged). The model recognises motif *shape*; it cannot yet tell '
        'lawful instances of the same shape from planted ones.',
        f'3. **Gather-scatter (relay hub) FPs are numerous but cheap.** {int(t1250.set_index("Type").loc["Gather-scatter (relay hubs)", f"FP: normal in the top {k:,}"])} '
        f'are in the top 1,250, but only {int(top_budget_fp["Gather-scatter (relay hubs)"])} are in the top {budget}. '
        f'The costly top-{budget} FPs come from layered stack ({int(top_budget_fp["Layered stack"])}), scatter-gather ({int(top_budget_fp["Scatter-gather"])}) '
        f'and bipartite ({int(top_budget_fp["Bipartite"])}) look-alikes.',
        f'4. **The hard tier is almost entirely missed**: `aggregate_relay` and `distributed_relay` place 4 of 200 rows in the top 1,250; median ranks are about 12,000–14,000. '
        'The 25 ambiguous rows are unlearnable by design.',
        f'5. **Medium is the reachable gap**: {int((medium["rank"] <= k).sum())} of {len(medium)} medium rows reach the top 1,250 (median rank {int(medium["rank"].median()):,}). '
        'Whole medium scenarios sit at ranks 12,000–28,000, i.e. the model treats them as background.',
        f'6. **{partly_missed} of the {len(fn)} misses are in scenarios where at least one other leg was caught**; {fully_missed} are in scenarios with no caught leg. '
        'Scenario-context features (connected legs\' first-stage scores, built with nested folds) target the first group.',
        '7. **Easy families that recall well but rank low**: bipartite, stack, random and fan-out easy rows mostly reach the top 1,250 but rarely the top 402 '
        '(see table 3), so they sit just under the look-alike FPs. Reordering easy rows above look-alikes is the largest AP lever with existing signal.', '']
    show = ['rank', 'score', 'type', 'family', 'scenario_id', 'hours_from_start', 'FROM_ADDRESS', 'TO_ADDRESS', 'VALUE',
            'execution_legs', 'pair_count', 'flow_300s_two_sided_count', 'TRANSACTION_HASH']

    lines = [
        '# V1 validation: false positives and false negatives by type', '',
        '**Source.** V1 graph CatBoost, scenario/cohort-grouped 3-fold out-of-fold scores (`outputs/strategy3_v1/graph_oof.csv`) '
        f'on all {len(d):,} labelled training transactions ({k:,} suspicious). Pooled AP = **{ap:.4f}** (score {100 * ap:.1f}). '
        'Local validation only — not the hidden test score.', '',
        f'**Ranking.** Score descending, ties broken by transaction hash ascending (portal convention). The cut-off is the top **{k:,}** '
        f'(equal to the number of suspicious rows, so precision = recall at this cut-off) and, separately, the portal review budget of the top **{budget}** (1%).', '',
        '**Type mapping.** Scatter-gather = `fan_out_in` + `multi_stage`; Layered stack = `stack` + `chain` + `distributed_relay` (hard); '
        'Gather-scatter = `aggregate_relay` (hard); Irregular network = `random`; Misc = `ambiguous` (labels independent of observables per the README). '
        'Annotation fields are used for reporting only.', '',
        *findings,
        f'## 1. Type table — top {k:,}', '', md(t1250), '',
        f'## 2. Type table — portal review budget, top {budget}', '', md(t402), '',
        '## 2b. Rank bands of 500', '',
        'Bands of 500 rows down to rank 5,000, then 5,001–10,000, 10,001–20,000 and the rest. '
        'Every 500-row band to the bottom is in `rank_bands_500_*.csv`.', '',
        '### Overview', '', md(band_summary), '',
        '### Suspicious rows (true positives) by type and band', '', md(band_sus), '',
        '### Normal rows (false positives) by type and band', '', md(band_nor), '',
        '## 3. Suspicious rows by family and difficulty', '',
        'Category AP ranks one category against all 38,881 legitimate rows; prevalence differs, so compare across categories with care.', '',
        md(fam, 3), '',
        '## 4. False positives (legitimate rows ranked in the top 1,250)', '',
        '### 4a. Where in the ranking they sit', '', 'FPs in the top 402 cost the most AP: they sit above most of the true positives.', '', md(fp_bands), '',
        '### 4b. Concentration: legitimate scenarios contributing the most FPs', '',
        f'{len(fp)} FP rows come from {fp.group.nunique()} distinct legitimate scenarios/background rows. '
        f'The 20 largest contributors account for {fp_scen.head(20).fp_rows.sum()} rows. '
        f'{int((wallets >= 5).sum())} wallets appear in five or more FP rows (max {int(wallets.max())}).', '',
        md(fp_scen.head(20).assign(group=lambda f: f.group.str[:16], median_rank=lambda f: f.median_rank.astype(int))), '',
        '### 4c. The 30 highest-ranked false positives', '',
        '`pair_count` = payments between this sender and receiver in the week; `flow_300s_two_sided_count` = matching inflow/outflow within 5 minutes.', '',
        md(short(top_fp[show]), 4), '',
        '## 5. False negatives (suspicious rows ranked below 1,250)', '',
        '### 5a. How far down they sit', '', md(fn_bands), '',
        '### 5b. Scenario-level view: are whole scenarios missed, or only some legs?', '',
        'A partly caught scenario has at least one row in the top 1,250; its missed legs are reachable through context from the caught legs. '
        'A fully missed scenario has no foothold.', '', md(scen), '',
        '### 5c. The 30 lowest-ranked easy/medium positives (hard tier excluded)', '',
        md(short(worst[show[:4] + ['difficulty'] + show[4:]]), 4), '',
        '## Files', '',
        '- `type_table_top1250.csv`, `type_table_top402.csv`: tables 1 and 2',
        '- `family_difficulty_table.csv`: table 3',
        '- `false_positives_top1250.csv`: every FP row with context', '- `false_positive_scenarios.csv`: FP concentration by scenario',
        '- `false_negatives_below1250.csv`: every FN row with context', '- `suspicious_scenarios.csv`: catch status per suspicious scenario', '',
        'All figures are local out-of-fold diagnostics; verify before using them for decisions or presentation.', '']
    (OUT / 'FP_FN_Report.md').write_text('\n'.join(lines))
    print(f'AP {ap:.4f}  K={k} budget={budget}  FP={len(fp)} FN={len(fn)}')


if __name__ == '__main__':
    main()
