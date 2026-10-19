import numpy as np
from cellpert.data import make_synthetic_dataset, save_response_npz, load_response_npz
from cellpert.models import ResponsePredictor
from cellpert.metrics import response_metrics

def test_synthetic_shapes():
    d=make_synthetic_dataset(seed=1,n_perturbations=20,n_genes=40,embedding_dim=8)
    assert d["delta"].shape==(20,40)
    assert d["embeddings"].shape==(40,8)

def test_npz_roundtrip(tmp_path):
    p=tmp_path/"r.npz"
    save_response_npz(p,["g1","g2"],["a","b"],[[1,2],[3,4]])
    d=load_response_npz(p)
    assert d["delta"].shape==(2,2)

def test_ridge_predict():
    rng=np.random.default_rng(2); X=rng.normal(size=(30,8)); Y=rng.normal(size=(30,10))
    m=ResponsePredictor("ridge_gene").fit(X[:24],Y[:24])
    assert m.predict(X[24:]).shape==(6,10)

def test_metrics_perfect():
    y=np.array([[1.,2.,3.],[2.,3.,4.]])
    m=response_metrics(y,y)
    assert m["mse"]==0
    assert abs(m["pcc_delta"]-1)<1e-8
    assert m["deg_recovery_top20"]==1
