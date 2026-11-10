"""Offline tests with generated, schema-compatible fixtures."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data import load_uci_csv, data_audit, MODEL_FEATURES, EXCLUDED_FEATURES
from src.modeling import profile_disjoint_holdout, compare_candidates
from src.evaluation import (
    capacity_metrics, operating_point, probability_metrics, select_f2_threshold,
    stratified_bootstrap_intervals, profile_bootstrap_intervals,
)
from src.modeling import make_candidates


def make_schema_fixture(n=200, seed=42):
    """Fully invented values used ONLY for deterministic software tests."""
    rng = np.random.default_rng(seed)
    calls = rng.poisson(50, n)
    complaints = rng.binomial(1, .20, n)
    usage = rng.gamma(2.2, 1500, n).astype(int)
    status = rng.choice([1,2],n,p=[.8,.2])
    logits = -2.2 + 1.6*complaints + 1.1*(status==2) - .004*calls
    y = rng.binomial(1,1/(1+np.exp(-logits)),n)
    # Extremely unlikely edge case: deterministically ensure two classes.
    if len(set(y))<2: y[:2]=[0,1]
    return pd.DataFrame({
        'Call Failure':rng.poisson(4,n),
        'Complains':complaints,
        'Subscription Length':rng.integers(3,48,n),
        'Charge Amount':rng.integers(0,11,n),
        'Seconds of Use':usage,
        'Frequency of use':calls,
        'Frequency of SMS':rng.poisson(25,n),
        'Distinct Called Numbers':rng.poisson(20,n),
        'Age Group':rng.integers(1,6,n),
        'Tariff Plan':rng.integers(1,3,n),
        'Status':status,
        'Age':rng.integers(18,65,n),
        'Customer Value':np.round(rng.uniform(10,1500,n),2),
        'Churn':y,
    })


def test_schema_and_content_validation():
    original = make_schema_fixture()
    with TemporaryDirectory() as temp:
        f = Path(temp)/'dataset.csv'
        original.to_csv(f,index=False)
        parsed=load_uci_csv(f,strict_rows=False)
        assert parsed.shape==(200,14)
        audit=data_audit(parsed)
        assert audit['rows']==200 and audit['missing_values']==0
        with pytest.raises(ValueError,match='3,150'):
            load_uci_csv(f,strict_rows=True)
        original.drop(columns=['Status']).to_csv(f,index=False)
        with pytest.raises(ValueError,match='Missing'):
            load_uci_csv(f,strict_rows=False)
        original.iloc[:,:].assign(Churn=2).to_csv(f,index=False)
        with pytest.raises(ValueError,match='Churn'):
            load_uci_csv(f,strict_rows=False)


def test_predictors_exclude_leakage_and_demographics():
    assert len(MODEL_FEATURES)==9
    assert not set(MODEL_FEATURES).intersection(set(EXCLUDED_FEATURES) | {'Churn'})
    fixture = make_schema_fixture(n=220)
    x=fixture[list(MODEL_FEATURES)]
    y=fixture['Churn']
    for name, model in make_candidates().items():
        model.fit(x,y)
        probs=model.predict_proba(x)[:,1]
        assert probs.shape==(len(x),) and ((probs>=0)&(probs<=1)).all(),name


def test_rank_metrics_and_f2_threshold_are_consistent():
    y=np.array([1,0,0,1,1,0,0,1,0,0])
    scores=np.array([.92,.11,.42,.88,.82,.18,.29,.74,.52,.23])
    cut=select_f2_threshold(y,scores)
    p=operating_point(y,scores,cut)
    assert 0<=p['f2']<=1
    assert p['threshold']==cut
    c=capacity_metrics(y,scores,fraction=.2)
    assert c['reviewed_count']==2 and c['churners_in_reviewed']==2
    assert c['precision_at_k']==1 and c['lift_at_k']==2.5
    m=probability_metrics(y,scores)
    assert m['roc_auc']>0.9 and m['average_precision']>0.9
    bounds=stratified_bootstrap_intervals(y,scores,n_resamples=35)
    assert all(0<=low<=high<=1 for low,high in bounds.values())
    with pytest.raises(ValueError):
        capacity_metrics(y,scores,fraction=0)
    with pytest.raises(ValueError):
        probability_metrics(y,[.2]*9+[2.0])


def test_offline_downloader_validates_before_saving(monkeypatch, tmp_path):
    """Intercept HTTP locally; do not depend on access to UCI in CI."""
    from io import BytesIO
    import zipfile
    from scripts import download_data

    fixture = make_schema_fixture(n=3150)
    destination = tmp_path / 'iranian_churn.csv'
    buffer = BytesIO()
    with zipfile.ZipFile(buffer,'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('Customer Churn.csv', fixture.to_csv(index=False))
    payload = buffer.getvalue()
    monkeypatch.setattr(download_data, 'DEST', destination)
    monkeypatch.setattr(download_data, 'urlopen', lambda request, timeout: BytesIO(payload))
    download_data.main()
    assert destination.exists()
    assert len(load_uci_csv(destination)) == 3150
    assert not (tmp_path / 'iranian_churn.csv.part').exists()


def test_offline_downloader_does_not_keep_invalid_csv(monkeypatch, tmp_path):
    from io import BytesIO
    import zipfile
    from scripts import download_data

    destination = tmp_path / 'iranian_churn.csv'
    archive = BytesIO()
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('Customer Churn.csv','WrongColumn,Churn\n123,1\n')
    monkeypatch.setattr(download_data,'DEST',destination)
    monkeypatch.setattr(download_data,'urlopen',lambda request, timeout: BytesIO(archive.getvalue()))
    with pytest.raises(ValueError,match='schema'):
        download_data.main()
    assert not destination.exists()
    assert not (tmp_path / 'iranian_churn.csv.part').exists()



def test_unicode_whitespace_in_uci_headers(tmp_path):
    """Actual UCI download may contain U+00A0 inside three headings."""
    fixture = make_schema_fixture(n=120)
    changed = fixture.rename(columns={
        'Call Failure': 'Call\u00a0 Failure',
        'Charge Amount': 'Charge\u00a0 Amount',
        'Subscription Length': 'Subscription\u00a0 Length',
    })
    path = tmp_path / 'uci_nbsp.csv'
    changed.to_csv(path, index=False)
    actual = load_uci_csv(path, strict_rows=False)
    assert list(actual.columns) == list(fixture.columns)
    pd.testing.assert_frame_equal(actual, fixture)


def test_unicode_whitespace_in_downloaded_zip(monkeypatch, tmp_path):
    """The end-to-end ZIP downloader accepts UCI's special-space headings."""
    from io import BytesIO
    from scripts import download_data
    import zipfile
    fixture = make_schema_fixture(n=3150).rename(columns={
        'Call Failure': 'Call\u00a0 Failure',
        'Charge Amount': 'Charge\u00a0 Amount',
        'Subscription Length': 'Subscription\u00a0 Length',
    })
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('Customer Churn.csv', fixture.to_csv(index=False))
    dest = tmp_path / 'iranian_churn.csv'
    monkeypatch.setattr(download_data, 'DEST', dest)
    monkeypatch.setattr(download_data, 'urlopen', lambda request, timeout: BytesIO(buffer.getvalue()))
    download_data.main()
    assert dest.exists()
    assert len(load_uci_csv(dest)) == 3150


def test_profile_disjoint_holdout_and_cv():
    fixture = make_schema_fixture(n=300, seed=37)
    # Simulate exact repeated customers and a conflicting-target feature profile.
    repeated = fixture.iloc[:50].copy()
    repeated.iloc[0, repeated.columns.get_loc('Churn')] = 1 - int(repeated.iloc[0]['Churn'])
    data = pd.concat([fixture, repeated], ignore_index=True)
    train, test, group_ids = profile_disjoint_holdout(
        data, features=MODEL_FEATURES, target='Churn', fraction=.2, seed=42,
    )
    assert set(group_ids[train]).isdisjoint(group_ids[test])
    assert len(train) + len(test) == len(data)
    assert data_audit(data)['conflicting_target_profiles'] >= 1
    small = data.iloc[train]
    result = compare_candidates(small[list(MODEL_FEATURES)], small['Churn'],
                                groups=group_ids[train], n_splits=3)
    assert len(result) == 4
    assert result['mean_cv_ap'].between(0, 1).all()
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for i, j in cv.split(small[list(MODEL_FEATURES)], small['Churn'], group_ids[train]):
        assert set(group_ids[train][i]).isdisjoint(group_ids[train][j])


def test_profile_bootstrap_metrics():
    labels = np.array([0,1,0,1,1,0,1,0,0,1])
    proba = np.array([.2,.8,.1,.91,.7,.4,.6,.14,.23,.77])
    groups = np.array([2,2,3,4,5,6,7,8,9,10])
    intervals = profile_bootstrap_intervals(labels, proba, groups,
                                             n_resamples=25, seed=42)
    assert set(intervals) == {'average_precision','roc_auc','brier_score'}
    assert all(0 <= lo <= hi <= 1 for lo,hi in intervals.values())
