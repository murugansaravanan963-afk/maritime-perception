"""Apply this session's explicitly inspected semantic labels. Never release automatically."""
import copy,json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
R=Path(r'D:/Projects/maritime-dataset');sys.path.insert(0,str(R/'scripts'))
from release_gate import canonical_hash
from render_review import render
from semantic_sheets import CLASSES
idx=[json.loads(s) for s in (R/'reports/people_semantic/index_0000.jsonl').read_text(encoding='utf8').splitlines()]
rs={r['id']:r for r in map(json.loads,(R/'staging/candidates/openimages_people.jsonl').read_text(encoding='utf8').splitlines())}
# Keys refer to inspected source proposals. New coordinates are explicit manual
# labels from original-resolution inspection, not automatic unions or clipping.
S={
9: {'manual':[(4,[234,150,425,665]),(4,[355,163,482,277]),(4,[649,156,853,285])], 'why':'Add missing foreground wader; refine the two partly submerged women to visible extent. Three distinct people, reflections excluded.'},
20:{'keep':{'p0':4},'why':'One person seated in ocean surf. Face/clothing hierarchy boxes are parts of the same person.'},
29:{'keep':{'p0':4},'why':'One waist-deep person. Related photographer session with ordinals52/64; keep author guard.'},
30:{'keep':{'p1':4},'why':'Single seated water person. Keep the tighter visible-extent Girl box; Woman box represents the same instance and extends into reflection, not another person.'},
31:{'keep':{'p0':4},'why':'One waist-deep waterfall-pool person; plants/rocks/waterfall are background.'},
32:{'keep':{'p0':4,'p1':4,'p2':4},'why':'Three distinct men standing in surf; preserve grayscale source photograph and all three native extents.'},
50:{'keep':{'p0':4},'why':'One person standing ankle-deep in continuous ocean surf; garments and their native hierarchy labels are not separate targets.'},
52:{'keep':{'p0':4},'why':'One waist-deep person; same capture series as29/64, author guard retained.'},
64:{'keep':{'p0':4},'why':'One waist-deep person; same capture series as29/52, author guard retained.'},
89:{'manual':[(2,[110,163,449,426]),(2,[601,388,974,579])],'why':'Two occupied surfboards afloat. Explicitly rebox visible boards with their riders; separate aboard people are not swimmer targets. Leashes/wave spray excluded.'},
90:{'keep':{'p0':4},'why':'One person partially submerged in lake water; body truncation at image boundary is genuine.'},
94:{'keep':{'p0':4},'why':'One diver physically in water; railing is permanent structure, no second target.'},
99:{'manual':[(4,[304,55,765,637])],'why':'One waist-deep person. Extend right edge to include the visible right arm omitted by native person box; no reflection.'},
100:{'manual':[(4,[58,247,568,647]),(4,[515,314,948,594])],'why':'Two people lying in shallow ocean water. Restore child right leg omitted by native box; retain separately visible mother, no reflection.'},
101:{'manual':[(4,[673,281,824,448]),(4,[565,436,634,544])],'why':'Two children in surf. Include rear child outstretched hands and visible lower leg; foreground child seated partly submerged.'},
106:{'keep':{'a0':5},'why':'One horse partly submerged in ocean, hence water animal/other_obstacle. Woman rides horse above water and is not person_in_water. Horse native box does not include rider head/torso.'},
113:{'manual':[(4,[76,62,550,432]),(4,[154,573,344,918])],'why':'Two children partly submerged in surf. Refine visible hand/body extents and remove excessive foam margin around foreground boy.'},
}
out=R/'reports/review_openimages_water_corrected';out.mkdir(exist_ok=True);patches=[];evidence=[]
for n,s in S.items():
 q=idx[n];base=rs[q['id']];assert canonical_hash(base)==q['candidate_record_sha256'];r=copy.deepcopy(base);boxes=[]
 for p in q['proposals']:
  if p['key'] in s.get('keep',{}): boxes.append({'class_id':s['keep'][p['key']],'xyxy':p['xyxy'],'reviewed_native_rows':p['native_rows']})
 for c,xy in s.get('manual',[]):boxes.append({'class_id':c,'xyxy':xy,'annotation_method':'explicit_manual_visible_extent_from_original_frame'})
 r['boxes']=boxes;r['unresolved_objects']=[];r['issues']=[];r['viewpoint']='shore'
 r['parent_record_sha256']=canonical_hash(base)
 page=R/f'reports/people_semantic/page_{n//4:04d}.jpg'
 r['corrections']=[{'kind':'full_frame_semantic_adjudication','reason':s['why'],'original_unresolved_context':base['unresolved_objects'],'native_labels_present':sorted({CLASSES[a['LabelName']] for a in base['native_annotations']}),'kept_source_proposals':s.get('keep',{}),'replacement_boxes':boxes,'inspection_evidence':str(page),'inspection_evidence_sha256':hashlib.sha256(page.read_bytes()).hexdigest()}]
 r['resolved_native_context']={'reason':s['why']+' Remaining native body-part/clothing labels are not separate maritime objects. Full frame inspected for all six target categories.'}
 r['checks_pending']=['corrected_overlay_confirmation','source_terms','recording_group_verification','near_duplicate_review','independent_split_assignment']
 patches.append({'id':r['id'],'parent_record_sha256':r['parent_record_sha256'],'corrected_record':r})
 evidence.append({'ordinal':n,**render(r,out/f'{n:04d}.jpg')})
p=R/'staging/corrections/openimages_water_20261023.jsonl'
if p.exists():raise FileExistsError('Refuse overwrite explicit correction ledger')
p.write_text(''.join(json.dumps(x)+'\n' for x in patches),encoding='utf8')
(out/'evidence.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in evidence),encoding='utf8')
print(json.dumps({'corrected_images':len(patches),'boxes':sum(len(p['corrected_record']['boxes']) for p in patches),'rendered':str(out),'release_approvals':0}))
