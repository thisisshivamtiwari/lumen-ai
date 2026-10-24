from pathlib import Path
import os,json,html
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
ROOT=Path(os.environ.get('LUMEN_WORKSPACE',Path.cwd()));O=ROOT/'outputs/v2_experiments';v=json.loads((O/'acceptance.json').read_text());d=pd.read_csv(O/'model_comparison.csv');c=pd.read_csv(O/'confirmation_results.csv');r=json.loads((O/'baseline_reproduction.json').read_text())
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodySmall',fontName='Helvetica',fontSize=9.5,leading=13,spaceAfter=8));styles['Title'].fontSize=23;styles['Title'].leading=28;styles['Title'].textColor=colors.HexColor('#142c44');styles['Heading2'].textColor=colors.HexColor('#164b66')
story=[]
def p(t,style='BodySmall'):story.append(Paragraph(t,styles[style]))
def h(t):p(t,'Heading2')
def table(rows,widths):
 t=Table([[Paragraph(html.escape(str(z)),styles['BodySmall']) for z in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#a2b5c3')),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,8))
p('Lumen / Strategy 3','Title');p('V2 experiment review - 24 October 2026')
h('Decision: '+('candidate passes local gates' if v['accepted_by_local_evidence'] else 'retain V1'))
p('Five predeclared alternatives were tested on exactly the original folds. No replacement is released unless the primary gain, consistency, confirmation and integrity criteria pass. No submission, upload or publication was made. No hidden-test or leaderboard score is known.')
h('V1 reproduced exactly')
p(f'Clean environment: Python 3.12.13, pinned dependencies, raw supplied inputs. Fresh execution took {r["runtime_seconds"]:.2f} seconds. The predictions CSV is byte-identical to V1. Mean fold AP is <b>54.4829%</b>; pooled OOF AP is <b>53.7186%</b>. These measure different quantities.')
rows=[['Track A candidate','Mean AP %','Pooled AP %','Delta pp']]
for _,z in d.iterrows():rows.append([z.experiment,f'{100*z.mean_fold_ap:.3f}',f'{100*z.pooled_ap:.3f}',f'{100*z.absolute_mean_ap_delta:+.3f}'])
table(rows,[215,85,90,85]);p('AP is average precision; pp means percentage points. The frozen finalist is '+v['candidate']+'. Candidate selection used mean fold AP, not pooled AP. Depth-only changes preserve observations; the 190-feature addition tests local diversity plus chronological value-compatible events.')
h('Primary acceptance gates')
p('Predeclared: at least 1.00 pp mean AP gain, improvement in at least two of three folds, positive paired group-bootstrap lower bound, positive confirmation-seed gain, and no greater than 2.00 pp regression on wallet/time diagnostics. Passing local gates would still require clean reproduction and valid outputs.')
p('Failed checks: '+', '.join(k for k,val in v['checks'].items() if not val)+'.')
story.append(PageBreak());p('Validation and uncertainty','Title')
h('What the tracks estimate')
table([['Track','Boundary and limitation'],['A - original','Cohort, else scenario, else background row grouping; fixed 3-fold SGKF seed 20261024. No group overlap, but all 700 wallets are shared across fit/evaluation. About 33% of validation directed pairs recur in fit.'],['B0 - entities','Fit and evaluation use disjoint supervised wallet endpoint halves. Cross-half edges have no supervised role. All permitted unlabelled graph context remains visible; this is not an independent graph or scenario split.'],['C - time','First five days of labels fit; last two days evaluate. Full seven-day unlabelled context is retained under the offline permission. This does not establish streaming or causal detection.'],['B1 - reserved','A second wallet partition was locked before fitting. Its metrics remain uninspected if the primary gain fails; its labels already entered Track A, so this is not an independent final test.'],['D - slices','Annotations are used only for analysis and split grouping. Common-negative difficulty slices have different prevalences, so AP values are not directly comparable across difficulties.']],[95,380])
rows=[['Check','V1 AP %','Finalist AP %','Delta pp']]
for track in ['B0','C','B1']:
 q=c[c.track.eq(track)].set_index('experiment')
 if len(q):rows.append([track,f'{100*q.loc["v1","ap"]:.3f}',f'{100*q.loc[v["candidate"],"ap"]:.3f}',f'{100*(q.loc[v["candidate"],"ap"]-q.loc["v1","ap"]):+.3f}'])
table(rows,[95,125,130,125])
p(f'Confirmation seed 20261025 mean AP delta: {100*v["confirmation_seed_mean_ap_delta"]:+.3f} pp. Paired group-bootstrap 95% AP-delta interval: [{100*v["paired_group_bootstrap_delta_95pct"][0]:+.3f}, {100*v["paired_group_bootstrap_delta_95pct"][1]:+.3f}] pp ({v["bootstrap_replicates"]} replicates). This conditions on fitted OOF predictions, excludes retraining uncertainty and is not proof of hidden-test gain.')
p('The graph has one connected component, so component grouping gives no useful holdout. Repeated fold selection, seven days of history, limited positive cohorts, lawful relay motifs and batch variations absent from training remain substantial risks.')
story.append(PageBreak());p('Data, errors and reproduction','Title')
h('Verified data and feature semantics')
p('Official package checksums, unique keys, transfer coverage, composite ledger joins, timestamps and fee units pass. Train has 40,131 payments; test has 40,101. Splits have disjoint hashes and customer wallets. Each conversion remains one customer event; executed LP legs supply actual debit information and are not independent laundering hops. Source debit uses the first executed leg, never SendMax.')
p('Chronological candidates require strictly earlier incoming value at the same customer wallet and outgoing actual-debit reference USD within the tolerance. They are event-compatibility candidates, not asset-level provenance or end-to-end fund conservation. Controlled fixtures check strict ordering, simultaneous-event exclusion and value mismatch. Raw-row shuffling and poisoned labels leave the added features unchanged.')
h('Errors and stress')
p('V1 difficulty-slice AP: easy 73.08%, medium 2.72%, hard 0.82%. The common negative set has 38,881 rows; positive counts are 800, 225 and 225 respectively. Counts, prevalence and conditional group-bootstrap intervals accompany the slices. Twenty-five hard positives deliberately have unobservable labels. Legitimate relay activity contributes high-scoring false positives; adding relay suspicion indiscriminately would hurt precision.')
p('All candidate difficulty comparisons are in candidate_difficulty_comparison.csv. Sparse-history stress removes 30% of label-blind economic events and recomputes context. confirmation_results.csv compares exactly the same retained evaluation targets under full and reduced history; inference shift is not confused with a retrained ablation. On 8,370 retained temporal targets, V1 AP falls from 56.81% to 41.35% and depth8 from 56.21% to 42.07%. Missing history is a major operational weakness; the competition itself supplies full windows.')
h('Reproducible deliverables')
p('README.md gives exact commands for a fresh, nonexistent workspace, pinned dependencies, baseline regeneration, fixed split audit, all five candidates and confirmation. Run parameters: 1,200 CatBoost iterations, learning rate 0.035, L2 10, rsm 0.85, depth 6 baseline / 4 and 8 variants, four CPU threads, seed 20261024; confirmation seed 20261025. Full results, runtimes, ablations and aggregate evidence accompany the source ZIP. V1 files are verified against their pre-experiment SHA256 manifest.')
p('Baseline predictions SHA256: '+r['original_sha256'])
h('Disclosure and release status')
p('Only supplied competition data was used. No external labels, test answers, hidden generator files, target propagation, pseudo labels, jitter or leaderboard tuning. OpenAI Codex assisted this work; the exact hidden model identifier and sampling settings are not exposed. The complete visible mission and available settings are recorded. Organiser helper/starter code retains its original rights.')
p('Public terms: https://hackathon.ukcbt.org/terms (inspected 24 October 2026). A future approved entry requires public original material by the deadline, repository version, team/captain/members and complete AI/resource disclosures. Those release actions remain pending user approval. This is a local research package, not organiser acceptance.')
def footer(canvas,doc):
 canvas.setStrokeColor(colors.HexColor('#d1dce4'));canvas.line(42,39,A4[0]-42,39);canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#536778'));canvas.drawString(42,26,'Lumen | Local evidence only | 24 October 2026');canvas.drawRightString(A4[0]-42,26,str(doc.page))
SimpleDocTemplate(str(O/'methodology.pdf'),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=48).build(story,onFirstPage=footer,onLaterPages=footer)
print('Wrote methodology.pdf')
