"""Record this session's actual visual findings; no release attestations generated."""
import sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(r'D:\Projects\maritime-dataset');sys.path.insert(0,str(ROOT/'scripts'))
from release_gate import canonical_hash
# Every entry below corresponds to a full frame and all object detail crops
# actually inspected in this session. 'annotation_pass' is not release approval.
L={
1:('quarantine','Native Float is a ladder-equipped anchored swimming platform/dock, outside water-debris scope; revise this object rather than map Float blindly.'),
2:('annotation_pass','Navigation buoy and one distant water-supported person; no additional clear targets observed.'),
3:('annotation_pass','One swimmer and a yellow marker. Fixed swimming-lane barrier is structure; do not box reflections.'),
4:('annotation_pass','One visible swimmer; fixed lane barrier and shore marina are outside the external-water-object scope.'),
5:('annotation_pass','One visible swimmer; same bathing area as mastr_0811, link before splitting.'),
6:('annotation_pass','One swimmer and a nonmotorized canoe including its two occupants; reflection excluded.'),
7:('annotation_pass','Four water-supported swimmers and one canoe. Same river/participants as nearby inhouse 45-53 recordings.'),
8:('annotation_pass','One paddleboard with standing rider and one truncated swimmer at right; rider is not a swimmer label.'),
9:('quarantine','Crowded bathers at right are not represented by distinct complete swimmer boxes; instance separation/completeness unresolved.'),
10:('quarantine','Overlapping paddleboard instances and a swimmer box at board edge need reannotation; do not merge or delete by guesswork.'),
11:('quarantine','Native paddleboard is visibly sail-equipped windsurfing craft; incompatible direct small_craft mapping. Resolve consistently before admission.'),
12:('quarantine','Same sail-equipped board as preceding sequence frame; native paddleboard mapping is unresolved.'),
13:('quarantine','Same sail-equipped board as preceding sequence frame; native paddleboard mapping is unresolved.'),
14:('quarantine','Visible floating twig/debris in foreground has no target box. Six-class completeness not established.'),
15:('quarantine','Visible floating twig/debris near foreground has no target box. Six-class completeness not established.'),
16:('annotation_pass','One distant person physically in water; visible extent boxed. Capture timestamp 2019-08-10 links with inhouse104 event.'),
17:('annotation_pass','Three people wading/partly submerged at right boundary. People on dry sand remain outside scope.'),
18:('annotation_pass','Two people partly submerged; permanent breakwater and poles excluded.'),
19:('annotation_pass','Three partly submerged people; building on piles is permanent structure.'),
20:('annotation_pass','One water-supported person seen from a shore/deck camera. Rain and lens spots are image conditions, not file corruption.'),
21:('annotation_pass','One water-supported person in rain. Same site as inhouse112-115; prevent cross-split site leakage.'),
22:('quarantine','Distant vessel silhouette on horizon is not represented by a six-class target box; semantic identity unresolved.'),
23:('needs_review','Beach/waterline waders need native-resolution foot/waterline confirmation before certifying all four as in-water.'),
24:('quarantine','Overlapping person boxes at right appear to duplicate or fragment a person; instance identity unresolved.'),
25:('annotation_pass','One distant partly submerged person; breakwater and shore huts excluded.'),
26:('annotation_pass','One swimmer against a pier; person standing on the pier is excluded.'),
27:('annotation_pass','Two swimmers, four occupied nonmotorized boards and a marker. Aboard people remain part of craft boxes.'),
28:('annotation_pass','Two swimmer heads and one native buoy; no additional clear target visible.'),
29:('annotation_pass','One swimmer and two native buoys. Same scene/event as prior inhouse204 frame.'),
30:('annotation_pass','Two swimmers, three native buoys and a distant occupied nonmotorized craft; preserve the tiny boxes.'),
31:('annotation_pass','One foreground navigation/mooring buoy, two distant swimmers and a nonmotorized occupied craft at waterline.'),
32:('needs_review','Swimmer/canoe labels appear aligned, but floating foreground material needs scope/completeness review.'),
33:('annotation_pass','One swimmer and one occupied canoe; matches participants/location in inhouse45-48.'),
34:('annotation_pass','One clear swimmer. No additional clear six-class target observed.'),
35:('annotation_pass','One swimmer and two occupied canoes. Same river/location/participants as inhouse239/240.'),
36:('quarantine','Native Buoy box encloses a red occupied canoe visible also in adjacent frames. Correct class and verify full craft extent.'),
37:('annotation_pass','One swimmer and two occupied red canoes; same capture event as other inhouse240 frames.'),
38:('quarantine','Several distinct bathers are covered by a single crowded swimmer region; individual detection boxes incomplete.'),
39:('quarantine','At least one native swimmer box appears to include someone aboard a paddleboard/at waterline; resolve in-water status and overlapping instances.'),
40:('quarantine','Two distinct people share a swimmer box; crowded source mask cannot be treated as one detector instance.'),
41:('annotation_pass','Two swimmers and one red nonmotorized canoe; preserve the small distant swimmer.'),
42:('needs_review','Very small overlapping board instances require native-resolution separation and completeness confirmation.'),
43:('quarantine','Overlapping distant craft boxes and native marker/waterline-person ambiguity remain unresolved.'),
44:('quarantine','Distant swimmer boxes overlap ambiguously; marker-versus-person identity cannot be established confidently from this frame.'),
45:('annotation_pass','One swimmer in a river scene; source instance mask and visible body agree.'),
46:('needs_review','Board and swimmer align, but foreground floating material needs target-scope review.'),
47:('annotation_pass','Four people physically in water. Ball held by a person is not an unattached floating obstacle.'),
48:('needs_review','Tiny native buoy/person distinctions and occupied-craft extents need sequence confirmation; do not guess dark dots.'),
}
O={
1:('needs_review','Native white floating litter is correctly localized; an additional unboxed white speck to its left needs native-resolution identification.'),
2:('annotation_pass','One large floating plastic bag, tight native box; no other clear target observed.'),
3:('needs_review','Two boxed litter pieces; nearby floating fragments/twigs need broader water-obstacle scope review.'),
4:('quarantine','Several visible floating pieces are outside the three supplied litter boxes. Full-frame water-obstacle labels incomplete.'),
5:('needs_review','One litter piece localized correctly; other tiny floating material needs completeness confirmation.'),
6:('annotation_pass','One clear floating white litter piece; no other clear target observed.'),
7:('annotation_pass','One floating white litter piece; wall, wall fixtures and reflections excluded.'),
8:('quarantine','Native litter boxes omit multiple visible floating branches/logs in a debris accumulation; six-class completeness fails.'),
9:('annotation_pass','One partly transparent floating plastic piece boxed near left wall; permanent wall and own apparatus excluded.'),
10:('needs_review','Native transparent floating material box plausible; other foreground floating material requires full-resolution review.'),
11:('annotation_pass','One floating litter piece boxed near left wall. Scene fixtures and their reflections excluded.'),
12:('needs_review','White litter piece boxed; multiple other small floating pieces need completeness review.'),
13:('quarantine','A sizeable floating twig/stem is visible unboxed near frame center; add reviewed box or exclude whole image.'),
14:('annotation_pass','One transparent floating litter object; box visible in enlarged native crop.'),
15:('annotation_pass','One long transparent floating litter piece; no additional clear target observed.'),
16:('needs_review','Boxed litter visible, but other floating stems/material require completeness review.'),
17:('annotation_pass','One clearly boxed floating litter piece; no other clear target observed.'),
18:('needs_review','Floating bag is boxed, but additional floating debris specks need native-resolution review.'),
19:('needs_review','Boxed floating litter at left; other floating material at right requires scope/completeness review.'),
}
now=datetime.now(timezone.utc).isoformat();taxhash=hashlib.sha256((ROOT/'configs/taxonomy.json').read_bytes()).hexdigest();observations=[]
for source,dirname,decisions in [('lars','review_lars_swimmers',L),('oostpoort','review_oostpoort',O)]:
 candidates={r['id']:r for r in map(json.loads,(ROOT/f'staging/candidates/{source}.jsonl').read_text().splitlines())}
 evidence=list(map(json.loads,(ROOT/'reports'/dirname/'evidence.jsonl').read_text().splitlines()))
 assert len(evidence)==len(decisions)
 for n,(decision,note) in decisions.items():
  ev=evidence[n-1];r=candidates[ev['id']];assert canonical_hash(r)==ev['record_sha256']
  if r['issues'] and decision=='annotation_pass':decision='needs_review';note+=' Native structural/crowd issues also remain: '+str(r['issues'])
  ep=Path(ev['evidence_image'])
  observations.append({'id':r['id'],'record_sha256':canonical_hash(r),'taxonomy_sha256':taxhash,'image_sha256':r['image']['sha256'],'evidence_path':str(ep),'evidence_sha256':hashlib.sha256(ep.read_bytes()).hexdigest(),'reviewed_by':'Codex visual inspection; not a human annotation certification','reviewed_at':now,'inspection':'full-frame overlay plus every supplied object crop','decision':decision,'findings':note,'release_approved':False,'remaining_checks':['recording/site groups and cross-source overlap','near-duplicate/temporal review','frozen split allocation','final six-class review/signoff']})
out=ROOT/'staging/visual_observations.jsonl'
if out.exists():raise FileExistsError('Review ledger already exists; append a separately inspected batch, never overwrite prior observations')
out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in observations),encoding='utf8')
from collections import Counter
report={'state':'PARTIAL_VISUAL_REVIEW_NOT_DATASET_RELEASE','frames_inspected':len(observations),'decisions':dict(Counter(x['decision'] for x in observations)),'by_source':{s:dict(Counter(x['decision'] for x in observations if x['id'].startswith(s+':'))) for s in ['lars','oostpoort']},'release_approvals':0,'reviewer':'Codex visual inspection, not a human certification'}
(ROOT/'reports/visual_review_progress.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
