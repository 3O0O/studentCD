#!/usr/bin/env python3
"""Independent standard-library A audit; no model or student program execution.

Server mode rereads frozen source bodies in place to check selection, labels,
candidate hashes, score-derived probabilities and difflib metrics. Local mode
checks exported numeric evidence and byte hashes only. Neither mode refits q.
"""
import argparse
from collections import Counter, defaultdict
import difflib
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

METHODS = ('base', 'cd', 'b', 'history_d0')
Q_METHODS = ('overall_frequency', 'task_frequency', 'current_numeric', 'history_numeric', 'feedback_numeric')
POOLS = ('balanced12', '11_only12', 'legacy318', 'balanced12-history-sensitivity')
FORBIDDEN = {'code', 'current_code', 'target_code', 'predicted_code', 'raw_text',
             'generated_token_ids', 'completion_token_ids', 'history', 'feedback',
             'problem_statement', 'current_results', 'target_results'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(value):
    if isinstance(value, dict):
        for k, v in value.items():
            require(str(k).lower() not in FORBIDDEN, 'Private field in numeric artifact')
            safe(v)
    elif isinstance(value, list):
        for v in value:
            safe(v)
    elif isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite numeric artifact')


def unique(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, 'Duplicate JSON key')
        result[k] = v
    return result


def decode(raw):
    value = json.loads(raw, object_pairs_hook=unique)
    json.dumps(value, allow_nan=False)
    return value


def close(a, b, context='numeric value'):
    if a is None or b is None:
        require(a is b, 'Missing value differs: ' + context)
    else:
        require(type(a) in (int, float) and type(b) in (int, float)
                and math.isclose(a, b, rel_tol=2e-9, abs_tol=2e-10), 'Mismatch: ' + context)


def indexed(rows):
    result = {r['sample_id']: r for r in rows}
    require(len(result) == len(rows) and rows, 'Duplicate/empty sample cohort')
    return result


def mean(values):
    return statistics.mean(values) if values else None


def student_means(rows, key, subset=None):
    grouped = defaultdict(list)
    for r in rows:
        if r[key] is not None and (subset is None or bool(r['true_changed']) == subset):
            grouped[r['student_id']].append(r[key])
    return {s: mean(v) for s, v in sorted(grouped.items())}


def paired(a, b, key, n=2000, seed=20260929, subset=None):
    x, y = student_means(a, key, subset), student_means(b, key, subset)
    require(set(x) == set(y), 'Paired student coverage')
    ds = [x[s] - y[s] for s in sorted(x)]
    if not ds:
        return None, None
    rng = random.Random(seed)
    draws = sorted(mean([ds[rng.randrange(len(ds))] for _ in ds]) for _ in range(n))
    return mean(ds), [draws[int(.025*n)], draws[min(n-1, int(.975*n))]]


def sm(values):
    require(values and all(math.isfinite(v) for v in values), 'Finite scores required')
    vals = [math.exp(v-max(values)) for v in values]
    total = math.fsum(vals)
    return [v/total for v in vals]


def events(before, after):
    left, right = before.splitlines(keepends=True), after.splitlines(keepends=True)
    places, operations, size = set(), set(), 0
    for op, start, end, ns, ne in difflib.SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
        if op == 'equal':
            continue
        if op == 'insert':
            places.add(('boundary', start)); operations.add(('insert', start))
        else:
            places.update(('line', i) for i in range(start, end))
            operations.update((op, i) for i in range(start, end))
        size += end-start+ne-ns
    return places, operations, size


def body_metrics(current, predicted, target):
    p, po, ps = events(current, predicted)
    t, to, ts = events(current, target)
    def f1(a, b):
        return 1. if not a and not b else 2*len(a & b)/(len(a)+len(b))
    return {'exact_next_code': float(predicted == target), 'true_changed': float(current != target),
            'predicted_changed': float(predicted != current), 'edit_location_f1': f1(p, t),
            'edit_operation_f1': f1(po, to), 'edit_size_absolute_error': abs(ps-ts),
            'text_similarity': difflib.SequenceMatcher(None, predicted, target, autojunk=False).ratio()}


def w1(predicted, actual):
    a, b = defaultdict(float), defaultdict(float)
    for v, w in predicted: a[v] += w
    for v, w in actual: b[v] += w
    close(sum(a.values()), 1); close(sum(b.values()), 1)
    xs = sorted(set(a) | set(b))
    total, cdf = 0., 0.
    for i, x in enumerate(xs[:-1]):
        cdf += a[x]-b[x]
        total += abs(cdf)*(xs[i+1]-x)
    return total


def auc(pairs):
    pos = [(p, w) for y, p, w in pairs if y]
    neg = [(p, w) for y, p, w in pairs if not y]
    if not pos or not neg: return None
    numerator = math.fsum(w*v*(float(p>q)+.5*float(p==q)) for p, w in pos for q, v in neg)
    return numerator/(math.fsum(w for _, w in pos)*math.fsum(w for _, w in neg))


def verify_q(report, oof, protocol):
    rows = indexed(oof)
    require(len(rows) == report['sample_count'] and len({r['student_id'] for r in oof}) == report['student_count'], 'OOF coverage')
    require(report['default_method'] == 'feedback_numeric' and report['folds'] == 5
            and report['seed'] == 20261003 and report['l2'] == 1, 'Frozen q protocol')
    students = sorted({r['student_id'] for r in oof})
    shuffled = list(students); random.Random(20261003).shuffle(shuffled)
    assignment = {s: i % 5 for i, s in enumerate(shuffled)}
    require(report['fold_assignment'] == assignment, 'Label-independent fixed folds')
    require(report['changed_count']==sum(r['label_changed'] for r in oof)
            and report['unchanged_count']==len(oof)-report['changed_count'],'OOF class totals')
    for row in oof:
        require(row['fold'] == assignment[row['student_id']] and type(row['label_changed']) is int
                and row['label_changed'] in (0, 1), 'OOF fold/label')
        close(row['p_changed'], row['p_changed_by_method']['feedback_numeric'])
    for entry in report['fold_audit']:
        train, held = set(entry['train_student_ids']), set(entry['heldout_student_ids'])
        require(not train & held and train | held == set(students), 'Fold train/holdout separation')
        require(held == {s for s in students if assignment[s] == entry['fold']}, 'Fold heldout IDs')
        model = report['fold_models'][entry['fold']]
        require(set(model['training_student_ids']) == train, 'Fold model training IDs')
    comparisons = {}
    losses = {}
    for method in Q_METHODS:
        grouped = defaultdict(list)
        for row in oof:
            p = row['p_changed_by_method'][method]
            require(type(p) in (int, float) and 0 <= p <= 1, 'q probability bounds')
            grouped[row['student_id']].append((row['label_changed'], p))
        metric = report['metrics'][method]
        lossrows, student_values = [], defaultdict(list)
        weighted_confusion = {k: 0. for k in ('tn','fp','fn','tp')}
        confusion = Counter()
        for s, pairs in sorted(grouped.items()):
            c = Counter()
            for y, p in pairs:
                key = ('tp' if p >= .5 else 'fn') if y else ('fp' if p >= .5 else 'tn')
                c[key] += 1; confusion[key] += 1; weighted_confusion[key] += 1/len(pairs)
            ys = sum(y for y, _ in pairs); ns = len(pairs)-ys
            loss = mean([-y*math.log(min(1-1e-12,max(1e-12,p)))
                         -(1-y)*math.log1p(-min(1-1e-12,max(1e-12,p))) for y,p in pairs])
            bs = mean([(p-y)**2 for y,p in pairs])
            computed = {'logloss': loss, 'brier': bs, 'auc': auc([(y,p,1.) for y,p in pairs]),
                        'balanced_accuracy': .5*(c['tp']/ys+c['tn']/ns) if ys and ns else None,
                        'unchanged_false_edit': c['fp']/ns if ns else None,
                        'changed_miss': c['fn']/ys if ys else None}
            actual = next(r for r in metric['per_student'] if r['student_id'] == s)
            require(actual['sample_count']==len(pairs) and actual['changed_count']==ys
                    and actual['unchanged_count']==ns and all(actual['confusion'][k]==c[k] for k in ('tn','fp','fn','tp')),'Per-student q class counts')
            for key, value in computed.items():
                close(actual[key], value, 'q '+method+' '+key)
                if value is not None: student_values[key].append(value)
            lossrows.append({'student_id':s, 'true_changed':1., 'logloss':loss, 'brier':bs})
        for key in ('logloss','brier','auc','balanced_accuracy','unchanged_false_edit','changed_miss'):
            values=student_values[key]
            close(metric['student_macro'][key]['value'], mean(values), 'q macro '+key)
            require(metric['student_macro'][key]['eligible_students'] == len(values), 'q macro denominator')
            eligible=[s for s,pairs in grouped.items() if (key in ('logloss','brier')
                or (key in ('auc','balanced_accuracy') and {y for y,_ in pairs}=={0,1})
                or (key=='unchanged_false_edit' and any(y==0 for y,_ in pairs))
                or (key=='changed_miss' and any(y==1 for y,_ in pairs)))]
            close(metric['student_macro'][key]['student_coverage'],len(eligible)/len(students))
            require(metric['student_macro'][key]['eligible_samples']==sum(len(grouped[s]) for s in eligible),'Eligible q sample denominator')
        for key in weighted_confusion:
            close(metric['student_weighted_confusion'][key], weighted_confusion[key])
            require(metric['submission_confusion'][key] == confusion[key], 'q confusion')
        weighted_pairs = [(y,p,1/len(grouped[s])) for s in grouped for y,p in grouped[s]]
        close(metric['pooled_equal_student_auc'], auc(weighted_pairs), 'q pooled weighted AUC')
        tn,fp,fn,tp=(weighted_confusion[k] for k in ('tn','fp','fn','tp'))
        close(metric['pooled_equal_student_balanced_accuracy'],.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None)
        close(metric['submission_unchanged_false_edit'],confusion['fp']/(confusion['tn']+confusion['fp']) if confusion['tn']+confusion['fp'] else None)
        close(metric['submission_changed_miss'],confusion['fn']/(confusion['tp']+confusion['fn']) if confusion['tp']+confusion['fn'] else None)
        require(metric['sample_count']==len(oof) and metric['student_count']==len(students)
                and metric['changed_count']==report['changed_count'] and metric['unchanged_count']==report['unchanged_count'],'Probability evaluation class totals')
        bins = metric['reliability_5_bins']
        total_ece = 0.
        for i, b in enumerate(bins):
            selected = [(s,y,p,1/len(grouped[s])) for s in grouped for y,p in grouped[s] if min(4,int(p*5)) == i]
            wt = math.fsum(r[3] for r in selected)
            close(b['student_weight_share'], wt/len(students))
            require(b['sample_count'] == len(selected) and b['student_count'] == len({r[0] for r in selected}), 'Reliability counts')
            mp = math.fsum(p*w for _,_,p,w in selected)/wt if wt else None
            my = math.fsum(y*w for _,y,_,w in selected)/wt if wt else None
            close(b['mean_p_changed'], mp); close(b['observed_changed_fraction'], my)
            if wt: total_ece += wt/len(students)*abs(mp-my)
        close(metric['reliability_ece_equal_student'], total_ece)
        close(metric['reliability_sample_coverage'],1.)
        losses[method] = lossrows
    for method in Q_METHODS[1:]:
        for baseline in ('overall_frequency','task_frequency'):
            if method == baseline: continue
            comparisons[method+'_minus_'+baseline] = {}
            for key in ('logloss','brier'):
                diff, ci = paired(losses[method], losses[baseline], key,
                                  protocol['bootstrap']['repetitions'], protocol['bootstrap']['seed'])
                comparisons[method+'_minus_'+baseline][key] = {'difference':diff, 'student_paired_95_ci':ci}
    return comparisons


def verify_analysis(report, sealed, protocol, body=None, q_rows=None):
    n, seed = protocol['bootstrap']['repetitions'], protocol['bootstrap']['seed']
    require(report['samples'] == 70 and report['students'] == 17, 'Full dev coverage')
    pre = {(r['sample_id'],r['method']):r for r in sealed}
    require(len(pre) == 560 == len(report['distributions']), 'All 70 x 8 sealed distributions')
    require(len({(r['sample_id'],r['method']) for r in report['distributions']})==560,'Unique analysis distribution cube')
    rows_by = defaultdict(list)
    q_index = indexed(q_rows) if q_rows is not None else None
    candidate_sets = {}
    body_cache = {}
    for d in report['distributions']:
        sid, method = d['sample_id'], d['method']
        old = pre[sid,method]
        for key in ('student_id','q_changed','candidate_ids','groups','probabilities'):
            require(d[key] == old[key], 'Pre-label distribution altered')
        ps, ids, groups = d['probabilities'], d['candidate_ids'], d['groups']
        if q_index is not None:
            require(sid in q_index and q_index[sid]['student_id'] == d['student_id'], 'Frozen q sample/student alignment')
            close(d['q_changed'],q_index[sid]['p_changed'],'Frozen q artifact alignment')
        signature=(ids,groups,d['candidate_edit_sizes'],d['candidate_metrics'])
        if sid in candidate_sets:
            require(candidate_sets[sid] == signature,'Shared pool differs across methods')
        candidate_sets[sid]=signature
        scores, metrics, sizes = d['candidate_log_scores'], d['candidate_metrics'], d['candidate_edit_sizes']
        require(len(ids) == len(set(ids)) == len(ps) == len(scores) == len(metrics) == len(sizes), 'Candidate alignment')
        require(set(groups) == {'changed','unchanged'} and all(0<=p<=1 for p in ps), 'Both groups and valid probabilities')
        close(math.fsum(ps),1)
        expected_ps = sm(scores)
        if method.startswith('a_'):
            expected_ps = [0.]*len(ps)
            for group, mass in (('changed',d['q_changed']),('unchanged',1-d['q_changed'])):
                idx = [i for i,g in enumerate(groups) if g == group]
                for i,p in zip(idx,sm([scores[i] for i in idx])): expected_ps[i] = mass*p
        for p,q in zip(ps,expected_ps): close(p,q,'independent score normalization')
        actual = next(r for r in report['per_sample'][method] if r['sample_id'] == sid)
        truth = actual['true_changed']; require(truth in (0.,1.), 'Binary dev truth')
        if body:
            current, target, pool, raw_scores = body
            x,y = current[sid]['current_code'], target[sid]['target_code']
            require(set(ids)==set(pool[sid]), 'Complete original candidate pool')
            require(d['student_id'] == current[sid]['student_id'], 'Body student identity')
            close(truth,float(x!=y),'body true changed')
            for i,cid in enumerate(ids):
                code = pool[sid][cid]
                require(digest(code.encode()) == cid, 'Candidate body SHA')
                require(groups[i] == ('unchanged' if code == x else 'changed'), 'Independent body group')
                if (sid,cid) not in body_cache:
                    body_cache[sid,cid]=(body_metrics(x,code,y),events(x,code))
                calculated,ev = body_cache[sid,cid]
                for key,value in calculated.items(): close(metrics[i][key],value,'independent difflib '+key)
                require(d['candidate_edit_locations'][i] == [list(t) for t in sorted(ev[0])], 'Body edit locations')
                close(sizes[i],ev[2]); close(scores[i], raw_scores[sid,cid][method.removeprefix('a_')])
            require(d['true_edit_locations'] == [list(t) for t in sorted(events(x,y)[0])], 'True edit locations')
            require(d['current_line_count'] == len(x.splitlines(keepends=True)), 'Current line universe')
            close(actual['true_edit_size'],events(x,y)[2])
        measured = {}
        for key in metrics[0]:
            if key not in ('true_changed','predicted_changed'):
                measured['expected_'+key] = math.fsum(p*m[key] for p,m in zip(ps,metrics))
        cp = math.fsum(p for p,g in zip(ps,groups) if g == 'changed')
        selected = min(range(len(ids)),key=lambda i:(-ps[i],ids[i]))
        true_size = actual['true_edit_size']
        crps = math.fsum(p*abs(v-true_size) for p,v in zip(ps,sizes))
        crps -= .5*math.fsum(p*q*abs(v-w) for p,v in zip(ps,sizes) for q,w in zip(ps,sizes))
        universe = [('line',i) for i in range(d['current_line_count'])]+[('boundary',i) for i in range(d['current_line_count']+1)]
        target_locs = {tuple(t) for t in d['true_edit_locations']}
        predicted_locs = [{tuple(t) for t in locs} for locs in d['candidate_edit_locations']]
        brier = mean([(math.fsum(p for p,locs in zip(ps,predicted_locs) if loc in locs)-float(loc in target_locs))**2 for loc in universe])
        idx = [i for i,g in enumerate(groups) if g == ('changed' if truth else 'unchanged')]
        conditional = math.fsum(p*metrics[i]['edit_location_f1'] for i,p in zip(idx,sm([scores[i] for i in idx])))
        measured.update({'change_probability':cp,'change_brier':(cp-truth)**2,
            'change_log_loss_floor_1e12':-math.log(max(1e-12,cp if truth else 1-cp)),
            'log_loss_floor_used':float((cp if truth else 1-cp)<1e-12),
            'change_decision_error':float((cp>=.5)!=bool(truth)),
            'expected_edit_size':math.fsum(p*s for p,s in zip(ps,sizes)),
            'true_edit_size':true_size,'edit_size_crps':crps,'location_event_brier':brier,
            'false_edit_probability':cp if not truth else None,'missed_edit_probability':1-cp if truth else None,
            'argmax_changed':float(groups[selected]=='changed'),'argmax_edit_location_f1':metrics[selected]['edit_location_f1'],
            'observed_group_conditional_f1_DIAGNOSTIC':conditional,'oracle_f1_DIAGNOSTIC':max(m['edit_location_f1'] for m in metrics)})
        require(actual['argmax_candidate_id'] == old['argmax_candidate_id'] == ids[selected], 'Argmax/tie rule')
        for key,value in measured.items(): close(actual[key],value,key)
        rows_by[method].append({'sample_id':sid,'student_id':d['student_id'],'true_changed':truth,**measured})
    require(set(rows_by) == {m for x in METHODS for m in (x,'a_'+x)}, 'All eight methods')
    require(len(candidate_sets)==70 and sum(len(v[0]) for v in candidate_sets.values())==report['candidates'], 'Unique complete candidate count')
    for rows in rows_by.values():
        require(len(rows)==70 and {r['sample_id'] for r in rows}==set(candidate_sets)
                and len({r['student_id'] for r in rows})==17,'Per-method complete dev cohort')
    if q_index is not None: require(set(candidate_sets)==set(q_index), 'Full frozen q coverage')
    for method,rows in rows_by.items():
        for section,subset in (('all',None),('true_changed',True),('true_unchanged',False)):
            for key,summary in report['summary'][method][section].items():
                by = student_means(rows,key,subset)
                require(summary['students'] == len(by) and set(summary['per_student']) == set(by), 'Student summary coverage')
                close(summary['student_macro_mean'],mean(list(by.values())))
                for s,v in by.items(): close(summary['per_student'][s],v)
                values = [r[key] for r in rows if r[key] is not None and (subset is None or bool(r['true_changed'])==subset)]
                close(summary['submission_mean'],mean(values))
        counts = Counter(r['student_id'] for r in rows)
        predicted, actual = [], []
        for d in report['distributions']:
            if d['method'] != method: continue
            wt = 1/(len(counts)*counts[d['student_id']])
            predicted.extend((v,wt*p) for v,p in zip(d['candidate_edit_sizes'],d['probabilities']))
            r = next(r for r in rows if r['sample_id'] == d['sample_id'])
            actual.append((r['true_edit_size'],wt))
        close(report['summary'][method]['student_weighted_edit_size_wasserstein'],w1(predicted,actual))
    for comp,items in report['comparisons'].items():
        first,second = comp.split('_minus_')
        for key,item in items.items():
            target = 'observed_group_conditional_f1_DIAGNOSTIC' if key == 'changed_conditional_f1_DIAGNOSTIC' else key
            subset = True if key == 'changed_conditional_f1_DIAGNOSTIC' else None
            diff,ci = paired(rows_by[first],rows_by[second],target,n,seed,subset)
            close(item['difference'],diff,'bootstrap estimate')
            require((ci is None) == (item['student_paired_95_ci'] is None), 'Bootstrap availability')
            if ci:
                for a,b in zip(item['student_paired_95_ci'],ci): close(a,b,'bootstrap interval')
    return {'samples':70,'students':17,'distributions':560,'candidate_metrics_body_checked':bool(body)}


def verify(collection, protocol_path, body_sources=None):
    directory, protocol_path = Path(collection), Path(protocol_path)
    proto_bytes = protocol_path.read_bytes(); protocol = decode(proto_bytes)
    manifest_path = directory/'calculation-manifest.json'
    manifest = decode(manifest_path.read_bytes())
    require(manifest['protocol_sha256'] == digest(proto_bytes), 'Protocol original byte SHA')
    loaded = {}
    receipts = {}
    for name, expected in manifest['artifact_sha256'].items():
        require(Path(name).name == name and not (directory/name).is_symlink(), 'Unsafe artifact path')
        raw = (directory/name).read_bytes(); require(digest(raw)==expected,'Artifact original byte SHA')
        value = [decode(line) for line in raw.splitlines()] if name.endswith('.jsonl') else decode(raw)
        safe(value); loaded[name]=value; receipts[name]=expected
    for filename,expected_n,expected_s in (('train-cv.json',737,140),('history-train-cv.json',519,None)):
        require(loaded[filename]['sample_count']==expected_n,'Fixed q training cohort')
        if expected_s: require(loaded[filename]['student_count']==expected_s,'Fixed q student cohort')
    fit, prediction, distributions = (loaded[name] for name in ('q-fit-seal.json','q-prediction-seal.json','candidate-distribution-seal.json'))
    for seal in (fit,prediction,distributions):
        require(seal['protocol_sha256']==digest(proto_bytes) and seal['dev_labels_read'] is False
                and seal['test_files_read'] is False,'Pre-label seal protocol/boundaries')
    require(prediction['q_fit_seal_sha256']==receipts['q-fit-seal.json']
            and distributions['q_prediction_seal_sha256']==receipts['q-prediction-seal.json'],'Seal byte graph')
    for field,seal in (('model_artifact_sha256',fit),('predictions_sha256',prediction),('pre_label_distribution_sha256',distributions)):
        for name,expected in seal[field].items(): require(receipts.get(name)==expected,'Sealed artifact original SHA')
    require(set(prediction['predictions_sha256'])=={'q.dev.jsonl','q-history-sensitivity.dev.jsonl'},'Both q predictions sealed')
    require(len(distributions['pre_label_distribution_sha256'])==4,'All four pools sealed')
    body_by = {}
    body_receipts = {}
    if body_sources is not None:
        root = Path(body_sources['project_root'])
        require(str(root)=='/data/zzm110186486/projects/student-sim-cd' and root.resolve()==root,'Fixed body source root')
        def read(rel,field):
            path=root/rel; require(path.resolve()==path and not path.is_symlink(),'Canonical body source')
            raw=path.read_bytes(); require(digest(raw)==protocol[field],'Frozen body source byte SHA')
            body_receipts[str(rel)]={'sha256':digest(raw),'bytes':len(raw)}
            return [decode(line) for line in raw.splitlines()]
        train=read('data/prepared/progfeed-v2/inputs.train.jsonl','train_inputs_sha256')
        train_labels=indexed(read('data/prepared/progfeed-v2/labels.train.jsonl','train_labels_sha256'))
        current=indexed(read('data/prepared/progfeed-selected-v1/matched/inputs.dev.jsonl','dev_inputs_sha256'))
        target=indexed(read('data/prepared/progfeed-selected-v1/matched/labels.dev.jsonl','dev_labels_sha256'))
        splitraw=(root/'data/prepared/progfeed-v2/splits.json').read_bytes()
        require(digest(splitraw)==protocol['splits_sha256'],'Frozen student registry original SHA')
        registry=decode(splitraw)['students']
        require(all(registry[r['student_id']]=='train' for r in train)
                and all(registry[r['student_id']]=='dev' for r in current.values())
                and not ({r['student_id'] for r in train}&{r['student_id'] for r in current.values()}),'Independent train/dev student separation')
        auditraw=(root/'data/prepared/progfeed-v2/audit.json').read_bytes()
        require(digest(auditraw)==protocol['source_audit_sha256'],'Frozen source inventory byte SHA')
        inventory=decode(auditraw)['source_files']
        require(digest(json.dumps(inventory,ensure_ascii=False,sort_keys=True,allow_nan=False).encode())==protocol['source_inventory_fingerprint_sha256'],'Inventory fingerprint')
        scope=(protocol_path.parent/'progfeed-single-file-scope.json').read_bytes()
        require(digest(scope)==protocol['scope_sha256'],'Scope original byte SHA')
        pairs={(r['lab'],r['source_file']) for r in decode(scope)['pairs']}
        listings=defaultdict(list)
        for entry in inventory:
            parts=Path(entry['path']).parts
            if len(parts)>=5 and parts[0]=='all_labs' and entry['path'].endswith('.py'):
                listings['/'.join(parts[:4])].append('/'.join(parts[4:]))
        eligible=[]
        for r in train:
            prefix='all_labs/'+r['lab']+'/'+r['student_id']+'/'+r['current_timestamp']
            if ((r['lab'],r['source_file']) in pairs and any(f['text'].strip() for f in r['feedback'])
                and isinstance(r.get('problem_statement'),str) and r['problem_statement'].strip()
                and sorted(listings[prefix])==[r['source_file']]): eligible.append(r)
        require(len(eligible)==737 and len({r['student_id'] for r in eligible})==140,'Independent eligible full cohort')
        for filename,selected in (('train-oof.jsonl',eligible),('history-train-oof.jsonl',[r for r in eligible if r['history']])):
            x=indexed(selected); oof=indexed(loaded[filename]); require(set(x)==set(oof),'Independent input-only eligible IDs')
            for sid,r in x.items():
                require(oof[sid]['student_id']==r['student_id'] and oof[sid]['label_changed']==int(r['current_code']!=train_labels[sid]['target_code']),'Independent OOF truth')
        require(len(loaded['history-train-oof.jsonl'])==519,'Full history sensitivity cohort')
        for poolname in POOLS:
            source=poolname if poolname in protocol['pool_sources'] else 'balanced12'
            spec=protocol['pool_sources'][source]; base=root/spec['directory']
            pieces={}
            for name,field in (('manifest.json','manifest'),('candidates.jsonl','candidates'),('scores.jsonl','scores')):
                raw=(base/name).read_bytes(); require(digest(raw)==spec[field+'_sha256'],'Pool source original SHA')
                body_receipts[str(base/name)]={'sha256':digest(raw),'bytes':len(raw)}
                pieces[field]=decode(raw) if field=='manifest' else [decode(line) for line in raw.splitlines()]
            pools={r['sample_id']:{c['candidate_id']:c['code'] for c in r['candidates']} for r in pieces['candidates']}
            raw_scores={}
            for row in pieces['scores']:
                l00,l01,l10,l11=(row[k] for k in ('l00','l01','l10','l11'))
                d0,d1=l10-l00,l11-l01
                values={'base':l11,'cd':l11+d1,'b':l11+d1-d0,'history_d0':l11+d0}
                for method,value in values.items(): close(row['scores_weight_1'][method],value,'Four-branch fixed score')
                raw_scores[row['sample_id'],row['candidate_id']]=values
            body_by[poolname]=(current,target,pools,raw_scores)
        require(set(current)==set(target) and len(current)==70,'Full dev source cohort')
    qcomparisons = {}
    for prefix,cv,oof in (('primary','train-cv.json','train-oof.jsonl'),('history_sensitivity','history-train-cv.json','history-train-oof.jsonl')):
        qcomparisons[prefix]=verify_q(loaded[cv],loaded[oof],protocol)
    checks={}
    for pool in POOLS:
        qfile='q-history-sensitivity.dev.jsonl' if pool=='balanced12-history-sensitivity' else 'q.dev.jsonl'
        checks[pool]=verify_analysis(loaded['analysis.'+pool+'.json'],loaded['distributions.'+pool+'.pre-label.jsonl'],protocol,body_by.get(pool),loaded[qfile])
    require(manifest.get('test_files_read') is False and manifest.get('student_execution') is False
            and manifest.get('generation_performed') is False,'Static run declared boundaries')
    return {'schema_version':'student-sim-cd.a-static-independent-verification.v1','status':'verified',
            'scope':'server_source_body_and_numeric' if body_sources else 'numeric_and_declared_original_byte_hashes_only',
            'verifier_sha256':digest(Path(__file__).read_bytes()),'protocol_sha256':digest(proto_bytes),
            'calculation_manifest_sha256':digest(manifest_path.read_bytes()),'artifact_sha256':receipts,
            'source_body_verification':body_sources is not None,'body_sources':body_receipts,
            'q_comparisons':qcomparisons,'pool_checks':checks,'q_independently_refitted':False,
            'resource_release_verified':False,'limitations':['Does not refit q or independently establish optimizer correctness.',
                'Numeric-only mode does not see student source bodies; source claims are verified on the server.',
                'Runtime, job exit, budget settlement and absence of residual processes require separate acceptance.',
                'No execution-based correctness, learning benefit or test generalization is established.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',required=True,type=Path); p.add_argument('--protocol',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    try:
        report=verify(args.collection,args.protocol)
        raw=(json.dumps(report,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
        with args.output.open('xb') as f: f.write(raw)
    except Exception as error:
        print(json.dumps({'status':'failed','error_type':type(error).__name__,'message':str(error)})); return 1
    print(json.dumps({'status':report['status'],'scope':report['scope'],'output_sha256':digest(raw)})); return 0


if __name__=='__main__':
    raise SystemExit(main())
