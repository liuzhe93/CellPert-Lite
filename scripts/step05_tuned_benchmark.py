#!/usr/bin/env python3
"""Training-only tuning for CellPert-Lite. See --help for defaults.

Ridge: common inner KFold, select minimum mean validation MSE.
MLP: common inner holdout, explicit validation-MSE early stopping; select
alpha and epoch count there, then refit on ALL outer-training perturbations.
Outer test responses are used only after both models have been selected/refit.
Previous test sets were already inspected: this is an exploratory refinement,
not a previously untouched confirmatory evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


REPS = ['GenePT_ada', 'GenePT_3large', 'UCE_input_tokens']


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def claim_config(path, config):
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise RuntimeError(f'Configuration/input/code changed: {path}. Choose a new --outdir.')
    else:
        dump(path, config)


def mse(y, p):
    import numpy as np
    # Same unweighted mean across perturbations and response genes.
    return float(np.mean(np.square(y.astype(np.float64)-p)))


def mlp_train(x, y, seed, alpha, epochs, learning_rate, batch_size,
              validation=None, patience=30, min_delta=1e-7):
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(128,64), solver='adam',
                         activation='relu', alpha=alpha, batch_size=batch_size,
                         learning_rate_init=learning_rate, early_stopping=False,
                         random_state=seed, shuffle=True, max_iter=1)
    best = None
    best_loss = float('inf')
    significant_best = float('inf')
    wait = 0
    history = []
    for epoch in range(1,epochs+1):
        model.partial_fit(x,y)
        if not np.isfinite(model.loss_):
            raise RuntimeError('Nonfinite MLP training loss')
        item = {'epoch':epoch, 'training_loss':float(model.loss_)}
        if validation is not None:
            xv,yv = validation
            loss = mse(yv,model.predict(xv))
            if not np.isfinite(loss):
                raise RuntimeError('Nonfinite validation MSE')
            item['validation_mse'] = loss
            if loss < best_loss:
                best_loss = loss
                best = {'epoch':epoch, 'validation_mse':loss}
            if significant_best-loss > min_delta:
                significant_best = loss
                wait = 0
            else:
                wait += 1
        history.append(item)
        if epoch == 1 or epoch % 25 == 0:
            print('[MLP EPOCH]',epoch,'alpha=',alpha,
                  'val_mse=',item.get('validation_mse','refit'),flush=True)
        if validation is not None and wait >= patience:
            break
    # During selection only the best epoch/MSE is needed. Final model is
    # separately trained on all outer-training rows for that epoch count.
    return model, best, history


def process_run(run, config, x, y_train, train_labels, test_labels,
                folds, inner_train, inner_val, args):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    run.mkdir(parents=True, exist_ok=True)
    claim_config(run/'run_config.json', config)
    if (run/'STEP03_VERIFIED.json').exists():
        needed=['metrics.csv','predictions.npz','model_bundle_ridge.joblib',
                'model_bundle_mlp.joblib','selection_summary.json','split_manifest.json']
        if not all((run/p).exists() for p in needed):
            raise RuntimeError(f'Completed marker but missing artifacts in {run}')
        print('[SKIP VERIFIED]',run,flush=True)
        return True
    # x contains ONLY outer-training rows during tuning/refit.
    candidate_dir=run/'tuning'; candidate_dir.mkdir(exist_ok=True)
    ridge_rows=[]
    for alpha in args.ridge_alphas:
        for fold,(fit_idx,val_idx) in enumerate(folds):
            cached=candidate_dir/f'ridge_a{alpha:g}_fold{fold}.json'
            if cached.exists():
                record=json.loads(cached.read_text())
            else:
                start=time.perf_counter()
                pipe=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='cholesky'))
                pipe.fit(x[fit_idx], y_train[fit_idx])
                score=mse(y_train[val_idx],pipe.predict(x[val_idx]))
                if not np.isfinite(score):raise RuntimeError('Nonfinite Ridge validation MSE')
                record={'alpha':alpha,'fold':fold,'validation_mse':score,
                        'seconds':time.perf_counter()-start}
                dump(cached,record)
                del pipe
            ridge_rows.append(record)
            print('[RIDGE CV]',alpha,'fold=',fold,'MSE=',record['validation_mse'],flush=True)
    cv=pd.DataFrame(ridge_rows)
    cv.to_csv(run/'ridge_cv_results.csv',index=False)
    ranked=cv.groupby('alpha',as_index=False).validation_mse.mean().sort_values(
        ['validation_mse','alpha'],ascending=[True,False])
    selected_alpha=float(ranked.iloc[0]['alpha'])
    mlp_rows=[]
    scaler=StandardScaler().fit(x[inner_train])
    xi=scaler.transform(x[inner_train]);xv=scaler.transform(x[inner_val])
    for alpha in args.mlp_alphas:
        cached=candidate_dir/f'mlp_a{alpha:g}.json'
        if cached.exists():
            record=json.loads(cached.read_text())
        else:
            start=time.perf_counter()
            model,best,history=mlp_train(
                xi,y_train[inner_train],args.current_seed,alpha,args.max_epochs,
                args.learning_rate,args.batch_size,validation=(xv,y_train[inner_val]),
                patience=args.patience,min_delta=args.min_delta)
            record={'alpha':alpha,'best_epoch':best['epoch'],
                    'validation_mse':best['validation_mse'],
                    'epochs_run':len(history),'hit_epoch_limit':len(history)==args.max_epochs,
                    'seconds':time.perf_counter()-start}
            pd.DataFrame(history).to_csv(candidate_dir/f'mlp_a{alpha:g}_curve.csv',index=False)
            dump(cached,record)
            del model
        mlp_rows.append(record)
        print('[MLP VALIDATION]',record,flush=True)
    del xi,xv,scaler
    pd.DataFrame(mlp_rows).to_csv(run/'mlp_validation_results.csv',index=False)
    selected=sorted(mlp_rows,key=lambda r:(r['validation_mse'],-r['alpha']))[0]
    selection={'selection_metric':'validation MSE (lower is better)',
               'ridge_alpha':selected_alpha,
               'ridge_cv_mean_mse':float(ranked.iloc[0]['validation_mse']),
               'ridge_selected_at_grid_boundary': selected_alpha in
                    [min(args.ridge_alphas),max(args.ridge_alphas)],
               'mlp_alpha':selected['alpha'],'mlp_epochs':selected['best_epoch'],
               'mlp_alpha_selected_at_grid_boundary':selected['alpha'] in
                    [min(args.mlp_alphas),max(args.mlp_alphas)],
               'mlp_validation_mse':selected['validation_mse'],
               'mlp_selected_candidate_hit_epoch_limit':selected['hit_epoch_limit'],
               'mlp_hidden_layers':[128,64],
               'refit':'All outer-training rows; epoch count from inner validation',
               'outer_test_used_for_selection':False}
    dump(run/'selection_summary.json',selection)
    # No outer-test X or Y is passed to this function.
    start=time.perf_counter()
    ridge=make_pipeline(StandardScaler(),Ridge(alpha=selected_alpha,solver='cholesky'))
    ridge.fit(x,y_train)
    ridge_seconds=time.perf_counter()-start
    start=time.perf_counter()
    final_scaler=StandardScaler().fit(x)
    mlp,_,history=mlp_train(final_scaler.transform(x),y_train,args.current_seed,
                           selected['alpha'],selected['best_epoch'],args.learning_rate,args.batch_size)
    from sklearn.pipeline import Pipeline
    mlp=Pipeline([('standardscaler',final_scaler),('mlpregressor',mlp)])
    mlp_seconds=time.perf_counter()-start
    pd.DataFrame(history).to_csv(run/'mlp_refit_curve.csv',index=False)
    common={'genes':args.response_gene_names,'gene_embedding_dim':x.shape[1],
            'feature_mode':'gene_embedding_only','seed':args.current_seed,
            'training_perturbations':train_labels,'selection':selection}
    for label,model in [('ridge',ridge),('mlp',mlp)]:
        target=run/f'model_bundle_{label}.joblib'
        tmp=target.with_suffix('.tmp')
        joblib.dump({**common,'model':model,'model_name':label+'_gene'},tmp)
        tmp.replace(target)
    dump(run/'refit_timing.json',{'ridge_seconds':ridge_seconds,'mlp_seconds':mlp_seconds,
         'note':'Final-refit wall time only; not full tuning cost. Same thread budget.'})
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.')
    p.add_argument('--outdir',default='results/step05_tuned_genept_uce')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--ridge-alphas',type=float,nargs='+',default=[1,10,100,1000,10000])
    p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--mlp-alphas',type=float,nargs='+',default=[1e-4,1e-3,1e-2])
    p.add_argument('--inner-val-size',type=float,default=.2)
    p.add_argument('--max-epochs',type=int,default=400)
    p.add_argument('--patience',type=int,default=30)
    p.add_argument('--min-delta',type=float,default=1e-7)
    p.add_argument('--learning-rate',type=float,default=1e-3)
    p.add_argument('--batch-size',type=int,default=128)
    args=p.parse_args()
    if (args.threads<1 or args.inner_folds<2 or args.max_epochs<1 or args.patience<1
        or args.batch_size<1 or not 0<args.inner_val_size<1 or args.min_delta<0
        or args.learning_rate<=0):p.error('Invalid tuning arguments')
    for grid in [args.seeds,args.ridge_alphas,args.mlp_alphas]:
        if len(set(grid))!=len(grid):p.error('Duplicate seeds/grid values are not allowed')
    if any(a<=0 for a in args.ridge_alphas+args.mlp_alphas):p.error('Alphas must be positive')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:
        os.environ[key]=str(args.threads)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    # Import numeric libraries only after setting thread controls.
    import numpy as np
    import pandas as pd
    import sklearn
    import joblib
    from threadpoolctl import threadpool_limits
    from sklearn.model_selection import KFold,train_test_split
    root=Path(args.root).resolve()
    sys.path.insert(0,str(root))
    from cellpert.metrics import response_metrics
    import cellpert.metrics as metric_module
    out=root/args.outdir;out.mkdir(parents=True,exist_ok=True)
    # Prevent concurrent writers to the same output directory on Linux/HPC.
    import fcntl
    lock=(out/'step05.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another Step05 process is writing to this output directory')
    with threadpool_limits(limits=args.threads):
        source=root/'results/step03_genept_uce_common'
        rp=root/'data/processed/benchmark_genept_uce_common/response_matrix_common.npz'
        with np.load(rp,allow_pickle=False) as z:
            perts=z['perturbations'].astype(str);genes=z['genes'].astype(str);y=z['delta'].astype(np.float32)
        if y.shape!=(len(perts),len(genes)) or not np.isfinite(y).all():raise ValueError('Invalid response')
        if len(set(perts))!=len(perts) or len(set(genes))!=len(genes):raise ValueError('Duplicate response labels')
        lookup={g:i for i,g in enumerate(perts)}
        files={rep:source/'aligned_features'/f'{rep}.npz' for rep in REPS}
        inputs={'response':sha(rp),**{rep:sha(f) for rep,f in files.items()}}
        settings={k:v for k,v in vars(args).items() if k not in ['root','outdir','seeds']}
        settings.update({'input_sha256':inputs,'script_sha256':sha(Path(__file__).resolve()),
                         'metrics_sha256':sha(Path(metric_module.__file__)),
                         'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,
                         'python_version':sys.version,'selection_metric':'MSE',
                         'mlp_hidden_layers':[128,64],'exploratory_refinement':True})
        claim_config(out/'experiment_config.json',settings)
        # Freeze every requested split BEFORE training any representation.
        splits={}
        split_dir=out/'inner_splits';split_dir.mkdir(exist_ok=True)
        for seed in args.seeds:
            outer=json.loads((source/f'verified_split_seed_{seed}.json').read_text())
            if set(outer)!={'train','test'}:raise ValueError('Expected train/test outer split only')
            tr,te=outer['train'],outer['test']
            if (len(set(tr))!=len(tr) or len(set(te))!=len(te) or set(tr)&set(te)
                or set(tr)|set(te)!=set(perts)):raise ValueError('Invalid saved outer split')
            ti=np.array([lookup[g] for g in tr]);ei=np.array([lookup[g] for g in te])
            folds=list(KFold(args.inner_folds,shuffle=True,random_state=seed+1000).split(ti))
            it,iv=train_test_split(np.arange(len(ti)),test_size=args.inner_val_size,random_state=seed+2000)
            manifest={'seed':seed,'outer_train':tr,'outer_test':te,
                'ridge_folds':[{'train':[tr[i] for i in a],'validation':[tr[i] for i in b]} for a,b in folds],
                'mlp_inner_train':[tr[i] for i in it],'mlp_inner_validation':[tr[i] for i in iv]}
            claim_config(split_dir/f'seed_{seed}.json',manifest)
            splits[seed]=(ti,ei,folds,it,iv,outer)
        summaries=[];selected_rows=[];checks=[]
        for seed in args.seeds:
            ti,ei,folds,it,iv,outer=splits[seed]
            args.current_seed=seed;args.response_gene_names=genes
            yt=y[ti]
            for rep,path in files.items():
                start=time.perf_counter()
                with np.load(path,allow_pickle=False) as z:
                    if not np.array_equal(z['genes'].astype(str),perts):raise ValueError(f'Feature order mismatch: {rep}')
                    x=z['embeddings'].astype(np.float32)
                if x.ndim!=2 or len(x)!=len(perts) or not np.isfinite(x).all():raise ValueError(f'Invalid features: {rep}')
                run=out/f'seed_{seed}'/rep
                cfg={**settings,'seed':seed,'representation':rep,
                     'inner_split_sha256':sha(split_dir/f'seed_{seed}.json')}
                skipped=process_run(run,cfg,x[ti],yt,perts[ti],perts[ei],folds,it,iv,args)
                if not skipped:
                    print('[SELECTION COMPLETE; NOW EVALUATING OUTER TEST]',seed,rep,flush=True)
                    # Test outcomes first accessed here, after tuning and final refits.
                    predictions={'no_change':np.zeros((len(ei),len(genes)),np.float32),
                                 'train_mean':np.repeat(yt.mean(0)[None,:],len(ei),axis=0)}
                    for model,label in [('ridge_gene','ridge'),('mlp_gene','mlp')]:
                        bundle=joblib.load(run/f'model_bundle_{label}.joblib')
                        predictions[model]=bundle['model'].predict(x[ei]).astype(np.float32)
                    rows=[]
                    for model,pred in predictions.items():
                        if pred.shape!=y[ei].shape or not np.isfinite(pred).all():raise ValueError('Invalid final predictions')
                        rows.append({'model':model,**response_metrics(y[ei],pred)})
                    pd.DataFrame(rows).to_csv(run/'metrics.csv',index=False)
                    tmp=run/'predictions.tmp.npz'
                    np.savez_compressed(tmp,test_perturbations=perts[ei],genes=genes,y_true=y[ei],
                                        **{'pred_'+k:v for k,v in predictions.items()})
                    tmp.replace(run/'predictions.npz')
                    dump(run/'split_manifest.json',{'seed':seed,'n_train':len(ti),'n_test':len(ei),
                         'train_perturbations':perts[ti].tolist(),'test_perturbations':perts[ei].tolist()})
                    dump(run/'STEP03_VERIFIED.json',{'stage':'Step05 training-only tuning',
                         'config':cfg,'split':outer,'wall_seconds_this_attempt':time.perf_counter()-start})
                frame=pd.read_csv(run/'metrics.csv');frame.insert(0,'representation',rep);frame.insert(1,'seed',seed)
                summaries.append(frame)
                selection=json.loads((run/'selection_summary.json').read_text())
                selected_rows.append({'seed':seed,'representation':rep,**selection})
                checks.append({'seed':seed,'representation':rep,'n_train':len(ti),'n_test':len(ei)})
                print('[COMPLETE RUN]',seed,rep,flush=True)
                del x
        combined=pd.concat(summaries,ignore_index=True)
        combined.to_csv(out/'metrics_all_runs.csv',index=False)
        pd.DataFrame(selected_rows).to_csv(out/'selected_hyperparameters.csv',index=False)
        pd.DataFrame(checks).to_csv(out/'split_verification.csv',index=False)
        numeric=[c for c in combined.select_dtypes(include=np.number) if c!='seed']
        combined.groupby(['representation','model'])[numeric].agg(['mean','std','count']).to_csv(out/'metrics_mean_sd.csv')
        print('[STEP05 COMPLETE]',out,flush=True)
        print('Refits and validation tables saved; repeated holdouts are not independent cohorts.')
        print('Review selected_hyperparameters.csv for grid boundaries and epoch limits.')


if __name__=='__main__':
    main()
