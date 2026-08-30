"""Timing QC, constrained interpretable model fitting, and report data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable

from .manifest import CASES, manifest_document


PIPELINES = ("issue", "integer", "conversion", "fp32", "fp64", "special", "lut")
PENALTIES = ("dependency", "divergence", "spills")


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position)); upper = int(math.ceil(position))
    if lower == upper: return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize_samples(values: list[float], seed: int = 0x4F3C2D1A) -> dict[str, float]:
    median = statistics.median(values)
    q1, q3 = percentile(values, 0.25), percentile(values, 0.75)
    rng = random.Random(seed)
    bootstrap = [statistics.median(rng.choices(values, k=len(values))) for _ in range(2000)] if len(values) > 1 else [median]
    return {"median_ms": median, "q1_ms": q1, "q3_ms": q3, "iqr_ms": q3 - q1,
            "ci_low_ms": percentile(bootstrap, 0.025), "ci_high_ms": percentile(bootstrap, 0.975),
            "samples": len(values)}


def load_timings(path: Path) -> tuple[list[dict], dict]:
    with path.open(newline="") as handle: rows = list(csv.DictReader(handle))
    measurements: dict[str, list[float]] = {}
    anchors: dict[int, dict[str, float]] = {}
    for row in rows:
        value = float(row["per_dot_ms"])
        if row["stage"] == "measurement": measurements.setdefault(row["case_id"], []).append(value)
        elif "anchor" in row["stage"]:
            anchors.setdefault(int(row["round"]), {})[row["stage"]] = value
    expected = {case.case_id for case in CASES}
    if set(measurements) != expected:
        raise ValueError(f"timing cases mismatch missing={sorted(expected-set(measurements))} extra={sorted(set(measurements)-expected)}")
    summaries=[]
    for case in CASES:
        summary=summarize_samples(measurements[case.case_id], seed=0x4F3C2D1A ^ len(summaries))
        summary.update({"case_id":case.case_id,"split":case.split,"group":case.group})
        summaries.append(summary)
    drift=[]
    for round_index, pair in sorted(anchors.items()):
        if set(pair) != {"round_start_anchor","round_end_anchor"}: raise ValueError(f"incomplete anchors round {round_index}")
        start,end=pair["round_start_anchor"],pair["round_end_anchor"]
        drift.append(abs(end/start-1.0))
    noisy=[row for row in summaries if row["median_ms"] and row["iqr_ms"]/row["median_ms"]>0.02]
    qc={"anchor_round_drift":drift,"anchor_drift_max":max(drift,default=0.0),
        "anchor_drift_pass":max(drift,default=0.0)<=0.02,
        "noisy_case_count":len(noisy),"noisy_case_fraction":len(noisy)/len(summaries),
        "noise_pass":len(noisy)/len(summaries)<=0.02,
        "timing_pass":max(drift,default=0.0)<=0.02 and len(noisy)/len(summaries)<=0.02}
    return summaries,qc


def load_features(path: Path) -> dict[str, dict[str, float | str]]:
    with path.open(newline="") as handle: rows=list(csv.DictReader(handle))
    numeric=set(rows[0])-{"case_id","split","group","function"}
    result={}
    for row in rows:
        converted={key:(float(value) if key in numeric else value) for key,value in row.items()}
        result[str(row["case_id"])]=converted
    expected={case.case_id for case in CASES}
    if set(result)!=expected: raise ValueError("feature cases do not match manifest")
    return result


def raw_model_features(row: dict[str,float|str]) -> dict[str,float]:
    return {
        "issue":float(row["loop_instruction_count"]),
        "integer":float(row["integer_alu"])+2.0*float(row["integer_multiply"])+8.0*float(row["integer_divide"]),
        "conversion":float(row["conversion"]),
        "fp32":float(row["fp32"]),
        "fp64":float(row["fp64"]),
        "special":float(row["special"]),
        "lut":float(row["lut_expected_sectors_per_warp"])+float(row["global_loads"]),
        "dependency":float(row["critical_dependency_depth"])*(1.0-float(row["estimated_occupancy"])*0.5),
        "divergence":max(0.0,float(row["expected_true_warp_path"])+float(row["expected_false_warp_path"])-1.0)*float(row["predicated_instruction_count"]+row["branch_count"]),
        "spills":float(row["local_loads"])+float(row["local_stores"])+float(row["local_bytes_per_thread"])/8.0,
    }


def predict_vector(vector: dict[str,float], parameters: dict[str,float]) -> tuple[float,str]:
    pipeline={name:vector[name]*parameters[name] for name in PIPELINES}
    bottleneck=max(pipeline,key=pipeline.get)
    value=parameters["fixed"]+pipeline[bottleneck]+sum(vector[name]*parameters[name] for name in PENALTIES)
    return value,bottleneck


def huber_loss(errors: Iterable[float], delta: float) -> float:
    total=0.0
    for error in errors:
        absolute=abs(error)
        total += 0.5*error*error if absolute<=delta else delta*(absolute-0.5*delta)
    return total


def fit_model(summaries: list[dict], features: dict[str,dict]) -> dict:
    measured={row["case_id"]:row["median_ms"] for row in summaries}
    train_ids=[case.case_id for case in CASES if case.split=="train"]
    raw={case_id:raw_model_features(features[case_id]) for case_id in measured}
    scales={name:max(max(raw[case_id][name] for case_id in train_ids),1.0) for name in PIPELINES+PENALTIES}
    vectors={case_id:{name:value/scales[name] for name,value in values.items()} for case_id,values in raw.items()}
    baseline=min(measured[case_id] for case_id in train_ids)*0.8
    parameters={"fixed":baseline,**{name:baseline*0.2 for name in PIPELINES},**{name:baseline*0.05 for name in PENALTIES}}
    median_time=statistics.median(measured[case_id] for case_id in train_ids); delta=max(1e-6,median_time*0.03)
    names=["fixed",*PIPELINES,*PENALTIES]
    steps={name:max(median_time*0.2,1e-4) for name in names}
    def objective(candidate:dict[str,float])->float:
        return huber_loss((predict_vector(vectors[case_id],candidate)[0]-measured[case_id] for case_id in train_ids),delta)
    best=objective(parameters)
    for _ in range(250):
        changed=False
        for name in names:
            for direction in (1.0,-1.0):
                candidate=dict(parameters); candidate[name]=max(0.0,candidate[name]+direction*steps[name])
                score=objective(candidate)
                if score+1e-15<best: parameters,best,changed=candidate,score,True
        if not changed:
            for name in names: steps[name]*=0.6
            if max(steps.values())<1e-7: break
    predictions={case_id:predict_vector(vector,parameters) for case_id,vector in vectors.items()}
    train_errors=[measured[case_id]-predictions[case_id][0] for case_id in train_ids]
    residual90=percentile((abs(value) for value in train_errors),0.90)
    model={"schema_version":1,"form":"fixed + max(pipeline work * nonnegative cost) + additive nonnegative penalties",
           "fit_split":"train","fit_case_ids":train_ids,"parameters_ms":parameters,"feature_scales":scales,
           "interval_half_width_ms":residual90,"huber_delta_ms":delta,"objective":best,
           "pipeline_features":list(PIPELINES),"penalty_features":list(PENALTIES)}
    frozen=json.dumps(model,sort_keys=True,separators=(",",":")); model["frozen_sha256"]=hashlib.sha256(frozen.encode()).hexdigest(); model["frozen_before_final_evaluation"]=True
    return {"model":model,"vectors":vectors,"predictions":predictions}


def ranks(values:list[float])->list[float]:
    order=sorted(range(len(values)),key=values.__getitem__); result=[0.0]*len(values); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]: j+=1
        rank=(i+j-1)/2+1
        for k in range(i,j): result[order[k]]=rank
        i=j
    return result


def spearman(left:list[float],right:list[float])->float:
    a,b=ranks(left),ranks(right); am,bm=statistics.mean(a),statistics.mean(b)
    numerator=sum((x-am)*(y-bm) for x,y in zip(a,b)); denominator=math.sqrt(sum((x-am)**2 for x in a)*sum((y-bm)**2 for y in b))
    return numerator/denominator if denominator else 0.0


def evaluate(summaries:list[dict], fit:dict)->tuple[list[dict],dict]:
    measured={row["case_id"]:row for row in summaries}; half=fit["model"]["interval_half_width_ms"]
    rows=[]
    for case in CASES:
        predicted,bottleneck=fit["predictions"][case.case_id]; actual=measured[case.case_id]["median_ms"]
        rows.append({"case_id":case.case_id,"split":case.split,"group":case.group,"measured_ms":actual,
                     "predicted_ms":predicted,"interval_low_ms":max(0.0,predicted-half),"interval_high_ms":predicted+half,
                     "absolute_error_ms":abs(predicted-actual),"ape_percent":abs(predicted-actual)/actual*100.0,
                     "bottleneck":bottleneck})
    synth=[row for row in rows if row["split"]=="synthetic_validation"]; final=[row for row in rows if row["split"]=="final"]
    acceptance={
        "synthetic_validation_median_ape_percent":statistics.median(row["ape_percent"] for row in synth),
        "synthetic_validation_p90_ape_percent":percentile((row["ape_percent"] for row in synth),0.9),
        "final_real_median_ape_percent":statistics.median(row["ape_percent"] for row in final),
        "final_real_max_ape_percent":max(row["ape_percent"] for row in final),
        "final_real_spearman":spearman([row["measured_ms"] for row in final],[row["predicted_ms"] for row in final]),
    }
    acceptance["synthetic_validation_pass"]=(acceptance["synthetic_validation_median_ape_percent"]<=5.0 and acceptance["synthetic_validation_p90_ape_percent"]<=10.0)
    acceptance["final_real_pass"]=(acceptance["final_real_median_ape_percent"]<=7.5 and acceptance["final_real_max_ape_percent"]<=15.0 and acceptance["final_real_spearman"]>=0.9)
    acceptance["all_pass"]=acceptance["synthetic_validation_pass"] and acceptance["final_real_pass"]
    return rows,acceptance


def write_csv(rows:list[dict],path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def analyze(timing:Path,features_path:Path,output_dir:Path)->dict:
    summaries,qc=load_timings(timing); features=load_features(features_path); fit=fit_model(summaries,features)
    evaluations,acceptance=evaluate(summaries,fit)
    output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(summaries,output_dir/"timing_summary.csv"); write_csv(evaluations,output_dir/"predictions.csv")
    (output_dir/"model.json").write_text(json.dumps(fit["model"],indent=2,sort_keys=True)+"\n")
    result={"manifest":manifest_document(),"quality_control":qc,"acceptance":acceptance,
            "model_sha256":fit["model"]["frozen_sha256"]}
    (output_dir/"analysis.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result
